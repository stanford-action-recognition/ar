from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from args import get_args
from dataloaders.dataset import RGBDataset, FlowDataset
from network import C3D_model, R2Plus1D_model, R3D_model
from network.R2Plus1D_BERT import (
    rgb_r2plus1d_16f_34_bert10,
    rgb_r2plus1d_32f_34_bert10,
    rgb_r2plus1d_64f_34_bert10,
)

args = get_args()
HMDB_CLASS_NUM = 51
HMDB_SPLITS_DIR = "./fixtures/hmdb51_splits"
HMDB_RGB_DATASET_DIR = "./data/jpegs_256"
HMDB_FLOW_DATASET_DIR = "./data/tvl1_flow"
RGB_OUTPUT_DIR = "./data/rgb_output"
FLOW_OUTPUT_DIR = "./data/flow_output"
CLIP_LEN = 16

class StreamFusion(nn.Module):
    def __init__(self, stream_models, num_classes, device):
        super(StreamFusion, self).__init__()
        self.device = device
        self.stream_models = stream_models
        # Please make sure the last layer of each model is a nn.Linear :)
        self.input_dimension = sum([list(stream_model.modules())[-1].out_features for stream_model in stream_models])
        self.relu = nn.ReLU()
        self.fusion_layer = nn.Linear(in_features=self.input_dimension, out_features=num_classes, bias=True)

    def forward(self, inputs_list):
        assert len(self.stream_models) == len(inputs_list)
        outputs_list = []
        for i in range(len(inputs_list)):
            outputs = self.stream_models[i](inputs_list[i].to(self.device))
            outputs_list.append(outputs)
        merged_output = torch.cat(outputs_list, dim=1).to(self.device)
        fusion_output = self.fusion_layer(self.relu(merged_output))
        return fusion_output

class Train():
    def __init__(self):
        self.useTest = False
        self.test_interval = 20
        self.stream_configs = []
        self.criterion = nn.CrossEntropyLoss()

        with wandb.init(
                project="ar", entity="stanford-action-recognition", config=args
        ) as wb:
            self.wb = wb
            self.config = wb.config
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("config", self.config)
            print("device", self.device)

            streams = self.config.streams.split(",")
            for stream in streams:
                if stream in ["rgb", "flow"]:
                    stream_config = {}
                    stream_config["model_name"] = self.config.model
                    stream_config["dataset_name"] = stream
                    stream_config["optimizer_name"] = self.config.optimizer
                    self.stream_configs.append(stream_config)
                else:
                    print("We have not implement this stream.")
                    raise NotImplementedError

            self.initialize_models()
            self.train_val_sizes = {}
            self.train_val_sizes["train"], self.train_val_sizes["val"] = self.initialize_train_datasets()
            if self.config.is_toy:
                self.train_val_sizes["train"] = self.config.train_toy_size
                self.train_val_sizes["val"] = self.config.val_toy_size
            self.initialize_optimizers()

            for stream_config in self.stream_configs:
                stream_config["model"].to(self.device)
            self.stream_fusion = StreamFusion([stream_config["model"] for stream_config in self.stream_configs], num_classes=51, device=self.device).to(self.device)
            # for name, param in self.stream_fusion.named_parameters():
            #     print(name)  # only nn.ReLU and nn.Linear
            if self.config.optimizer == "SGD":
                self.stream_fusion_optimizer = optim.SGD(
                    self.stream_fusion.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=5e-4
                )
            else:
                self.stream_fusion_optimizer = optim.Adam(self.stream_fusion.parameters(), lr=self.config.lr)
            self.criterion.to(self.device)

    def initialize_models(self):
        for stream_config in self.stream_configs:
            num_channels = self.config.c3d_in_channel * 2 if stream_config["dataset_name"] == "flow" else 3
            if stream_config["model_name"] == "C3D":
                stream_config["model"] = C3D_model.C3D(
                    num_classes=HMDB_CLASS_NUM,
                    c3d_dropout_rate=self.config.c3d_dropout_rate,
                    in_channel=num_channels,
                    pretrained=False,
                )
                stream_config["train_params"] = [
                    {"params": C3D_model.get_1x_lr_params(stream_config["model"]), "lr": self.config.lr},
                    {"params": C3D_model.get_10x_lr_params(stream_config["model"]), "lr": self.config.lr * 10},
                ]
            elif stream_config["model_name"] == "R2Plus1D":
                stream_config["model"] = R2Plus1D_model.R2Plus1DClassifier(
                    num_classes=HMDB_CLASS_NUM, in_channel=num_channels, layer_sizes=(2, 2, 2, 2)
                )
                stream_config["train_params"] = [
                    {"params": R2Plus1D_model.get_1x_lr_params(stream_config["model"]), "lr": self.config.lr},
                    {
                        "params": R2Plus1D_model.get_10x_lr_params(stream_config["model"]),
                        "lr": self.config.lr * 10,
                    },
                ]
            elif stream_config["model_name"] == "R3D":
                stream_config["model"] = R3D_model.R3DClassifier(
                    num_classes=HMDB_CLASS_NUM, in_channel=num_channels, layer_sizes=(2, 2, 2, 2)
                )
                stream_config["train_params"] = stream_config["model"].parameters()
            elif stream_config["model_name"] == "R2Plus1D_BERT":
                # TODO: Integrate in_channel in models.
                stream_config["model"] = rgb_r2plus1d_16f_34_bert10(num_classes=HMDB_CLASS_NUM, in_channel=num_channels, length=16)
                stream_config["train_params"] = [
                    {"params": R2Plus1D_model.get_1x_lr_params(stream_config["model"]), "lr": self.config.lr},
                    {
                        "params": R2Plus1D_model.get_10x_lr_params(stream_config["model"]),
                        "lr": self.config.lr * 10,
                    },
                ]
            else:
                print("We have not implement this model.")
                raise NotImplementedError

    def initialize_train_datasets(self):
        sanity_check = {"train": set(), "val": set()}
        for stream_config in self.stream_configs:
            for split in ["train", "val"]:
                if stream_config["dataset_name"] == "rgb":
                    stream_config["%s_dataloader" % split] = DataLoader(
                        RGBDataset(
                            dataset_dir=HMDB_RGB_DATASET_DIR,
                            splits_dir=HMDB_SPLITS_DIR,
                            output_dir=RGB_OUTPUT_DIR,
                            dataset_percentage=self.config.dataset_percentage,
                            split=split,
                            clip_len=CLIP_LEN,
                        ),
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=self.config.num_workers,
                    )
                elif stream_config["dataset_name"] == "flow":
                    stream_config["%s_dataloader" % split] = DataLoader(
                        FlowDataset(
                            dataset_dir=HMDB_FLOW_DATASET_DIR,
                            splits_dir=HMDB_SPLITS_DIR,
                            output_dir=FLOW_OUTPUT_DIR,
                            in_channel=self.config.c3d_in_channel,
                            dataset_percentage=self.config.dataset_percentage,
                            split=split,
                            clip_len=CLIP_LEN,
                        ),
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=self.config.num_workers,
                    )
                else:
                    print("We have not implement this dataset.")
                    raise NotImplementedError
                sanity_check[split].add(len(stream_config["%s_dataloader" % split].dataset))
        assert len(sanity_check["train"]) == 1 and len(sanity_check["val"]) == 1
        return sanity_check["train"].pop(), sanity_check["val"].pop()

    def initialize_optimizers(self):
        for stream_config in self.stream_configs:
            if stream_config["optimizer_name"] == "SGD":
                stream_config["optimizer"] = optim.SGD(
                    stream_config["train_params"], lr=self.config.lr, momentum=0.9, weight_decay=5e-4
                )
            elif stream_config["optimizer_name"] == "Adam":
                stream_config["optimizer"] = optim.Adam(stream_config["model"].parameters(), lr=self.config.lr)
            else:
                print("Not supported optimizer.")
                raise NotImplementedError

    def train(self):
        for epoch in tqdm(range(0, self.config.epochs), desc='Epoch'):

            # each epoch has a training and validation step
            for phase in ["train", "val"]:
                # reset the running loss and corrects
                running_loss = 0.0
                running_corrects = 0.0

                # reset the iterator of datasets
                for i in range(len(self.stream_configs)):
                    self.stream_configs[i]["%s_dataloader_iter" % phase] = iter(self.stream_configs[i]["%s_dataloader" % phase])

                if phase == "train":
                    for stream_config in self.stream_configs:
                        stream_config["model"].train()
                    self.stream_fusion.train()
                else:
                    for stream_config in self.stream_configs:
                        stream_config["model"].eval()
                    self.stream_fusion.eval()

                for iteration in tqdm(range(self.train_val_sizes[phase]), desc='Iter'):
                    inputs_list = []  # list of inputs from all streams
                    for stream_config in self.stream_configs:
                        inputs, labels = next(stream_config["%s_dataloader_iter" % phase])
                        inputs_list.append(inputs.float().to(self.device))

                    outputs = self.stream_fusion(inputs_list)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    labels = labels.to(self.device)
                    loss = self.criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        # print("test grad:", self.stream_configs[0]["model"].res3d.conv4.block1.conv2.temporal_spatial_conv.weight.grad)
                        torch.nn.utils.clip_grad_norm_(
                            self.stream_fusion.parameters(), self.config.clip_max_norm
                        )
                        self.stream_fusion_optimizer.step()
                        for stream_config in self.stream_configs:
                            if stream_config["optimizer_name"] == "SGD":
                                torch.nn.utils.clip_grad_norm_(
                                    stream_config["train_params"], self.config.clip_max_norm
                                )
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    stream_config["model"].parameters(), self.config.clip_max_norm
                                )
                            stream_config["optimizer"].step()

                    running_loss += loss.item() * inputs_list[0].size(0)
                    if iteration % 1 == 0:
                        print("Iter loss:", loss.item() * inputs.size(0))

                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.train_val_sizes[phase]
                epoch_acc = running_corrects.double() / self.train_val_sizes[phase]

                if phase == "train":
                    self.wb.log(
                        {
                            "epoch": epoch,
                            "train_loss": epoch_loss,
                            "train_acc": epoch_acc,
                        },
                        step=epoch,
                    )
                else:
                    self.wb.log(
                        {
                            "epoch": epoch,
                            "val_loss": epoch_loss,
                            "val_acc": epoch_acc,
                        },
                        step=epoch,
                    )

                print(
                    "[{}] Epoch: {}/{} Loss: {} Acc: {}".format(
                        phase, epoch + 1, self.config.epochs, epoch_loss, epoch_acc
                    )
                )

if __name__ == "__main__":
    train = Train()
    train.train()

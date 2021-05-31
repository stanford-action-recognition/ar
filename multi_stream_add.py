from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from args import get_args
from dataloaders.dataset import RGBDataset, FlowDataset
from network import C3D_model, R2Plus1D_model, R3D_model
from network.R2Plus1D_BERT import (
    rgb_r2plus1d_16f_34_bert10,
    rgb_r2plus1d_32f_34_bert10,
    rgb_r2plus1d_64f_34_bert10,
)

# from datetime import datetime

args = get_args()
HMDB_CLASS_NUM = 51
HMDB_SPLITS_DIR = "./fixtures/hmdb51_splits"

HMDB_RGB_DATASET_DIR = "./data/jpegs_256"
HMDB_FLOW_DATASET_DIR = "./data/tvl1_flow"

RGB_OUTPUT_DIR = "./data/rgb_output"
FLOW_OUTPUT_DIR = "./data/flow_output"

PRETRAINED_MODEL_FORMAT = "./model/%s/%s_model.pt"

CLIP_LEN = 16

class Test():
    def __init__(self):
        self.useTest = False
        self.test_interval = 20
        self.stream_configs = []
        self.criterion = nn.CrossEntropyLoss()

        self.config = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("config", self.config)
        print("device", self.device)

        streams = self.config.streams.split(",")
        for stream in streams:
            if stream in ["rgb", "flow"]:
                stream_config = {}
                stream_config["model_name"] = self.config.model
                stream_config["dataset_name"] = stream
                self.stream_configs.append(stream_config)
            else:
                print("We have not implement this stream.")
                raise NotImplementedError

        self.initialize_models()
        self.train_val_sizes = {}
        self.train_val_sizes["train"], self.train_val_sizes["val"] = self.initialize_train_datasets()

        for stream_config in self.stream_configs:
            stream_config["model"].to(self.device)

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
                if self.config.use_pretrained:
                    stream_config["model"].load_state_dict(
                        torch.load(PRETRAINED_MODEL_FORMAT % (
                            stream_config["dataset_name"], stream_config["model_name"])))
                stream_config["model"].fc8 = nn.Linear(2048, 512)

            elif stream_config["model_name"] == "R2Plus1D":
                stream_config["model"] = R2Plus1D_model.R2Plus1DClassifier(
                    num_classes=HMDB_CLASS_NUM, in_channel=num_channels, layer_sizes=(2, 2, 2, 2)
                )
                if self.config.use_pretrained:
                    stream_config["model"].load_state_dict(
                        torch.load(PRETRAINED_MODEL_FORMAT % (
                            stream_config["dataset_name"], stream_config["model_name"])))
                stream_config["model"] = stream_config["model"].res2plus1d

            elif stream_config["model_name"] == "R3D":
                stream_config["model"] = R3D_model.R3DClassifier(
                    num_classes=HMDB_CLASS_NUM, in_channel=num_channels, layer_sizes=(2, 2, 2, 2)
                )
                if self.config.use_pretrained:
                    stream_config["model"].load_state_dict(
                        torch.load(PRETRAINED_MODEL_FORMAT % (
                            stream_config["dataset_name"], stream_config["model_name"])))
                stream_config["model"] = stream_config["model"].res3d

            elif stream_config["model_name"] == "R2Plus1D_BERT":
                # TODO: Integrate in_channel in models.
                # TODO: Change line 67 in R2Plus1D_BERT.py to remove the FC that maps features to classes.
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
            if self.config.freeze_stream_models:
                for param in stream_config["model"].parameters():
                    param.requires_grad = False

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

    def test(self):
        # each epoch has a training and validation step
        for phase in ["train", "val"]:
            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # reset the iterator of datasets
            for i in range(len(self.stream_configs)):
                self.stream_configs[i]["%s_dataloader_iter" % phase] = iter(self.stream_configs[i]["%s_dataloader" % phase])

            for stream_config in self.stream_configs:
                stream_config["model"].eval()

            num_batches = self.train_val_sizes[phase] // self.config.batch_size + \
                          (self.train_val_sizes[phase] % self.config.batch_size != 0)
            for iteration in tqdm(range(num_batches), desc='Iter'):
                outputs_list = []  # list of inputs from all streams

                should_continue = False
                for stream_config in self.stream_configs:
                    try:
                        inputs, labels = next(stream_config["%s_dataloader_iter" % phase])
                    except StopIteration:
                        should_continue = True
                        break
                    outputs = stream_config["model"](inputs.float().to(self.device))
                    outputs_list.append(outputs)

                if should_continue:
                    continue

                outputs = torch.sum(torch.stack(outputs_list), dim=0)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / self.train_val_sizes[phase]
            print("Accuracy of phase %s: %s", phase, epoch_acc)


if __name__ == "__main__":
    test = Test()
    test.test()

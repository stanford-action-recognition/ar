from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from args import get_args
from dataloaders.dataset import FlowDataset
from network import C3D_model, R2Plus1D_model, R3D_model

HMDB_SPLITS_DIR = "./fixtures/hmdb51_splits"

HMDB_RGB_DATASET_DIR = "./data/jpegs_256"
HMDB_FLOW_DATASET_DIR = "./data/tvl1_flow"
OUTPUT_DIR = "./data/flow_output"


def train_model():
    args = get_args()

    useTest = False  # See evolution of the test set when training
    test_interval = 20  # Run on test set every nTestInterval epochs

    with wandb.init(
        project="ar", entity="stanford-action-recognition", config=args
    ) as wb:
        config = wb.config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("config", config)
        print("device", device)

        if config.dataset == "HMDB51":
            num_classes = 51
            dataset_dir = HMDB_FLOW_DATASET_DIR
            splits_dir = HMDB_SPLITS_DIR
        else:
            print("We only implemented hmdb and ucf datasets.")
            raise NotImplementedError

        if config.model == "C3D":
            model = C3D_model.C3D(
                num_classes=num_classes,
                c3d_dropout_rate=config.c3d_dropout_rate,
                in_channel=config.c3d_in_channel * 2,
                pretrained=False,
            )
            train_params = [
                {"params": C3D_model.get_1x_lr_params(model), "lr": config.lr},
                {"params": C3D_model.get_10x_lr_params(model), "lr": config.lr * 10},
            ]
        elif config.model == "R2Plus1D":
            model = R2Plus1D_model.R2Plus1DClassifier(
                num_classes=num_classes,
                in_channel=config.c3d_in_channel * 2,
                layer_sizes=(2, 2, 2, 2)
            )
            train_params = [
                {"params": R2Plus1D_model.get_1x_lr_params(model), "lr": config.lr},
                {
                    "params": R2Plus1D_model.get_10x_lr_params(model),
                    "lr": config.lr * 10,
                },
            ]
        elif config.model == "R3D":
            model = R3D_model.R3DClassifier(
                num_classes=num_classes,
                in_channel=config.c3d_in_channel * 2,
                layer_sizes=(2, 2, 2, 2),
            )
            train_params = model.parameters()
        else:
            print("We only implemented C3D model.")
            raise NotImplementedError

        wb.watch(model)

        criterion = nn.CrossEntropyLoss()

        if config.optimizer == "SGD":
            optimizer = optim.SGD(
                train_params, lr=config.lr, momentum=0.9, weight_decay=5e-4
            )
        elif config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            print("Not supported optimizer.")
            raise NotImplementedError

        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        model.to(device)
        criterion.to(device)

        print("Training model on {} dataset...".format(config.dataset))
        train_dataloader = DataLoader(
            FlowDataset(
                dataset_dir=dataset_dir,
                splits_dir=splits_dir,
                output_dir=OUTPUT_DIR,
                in_channel=config.c3d_in_channel,
                dataset_percentage=config.dataset_percentage,
                split="train",
                clip_len=16,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_dataloader = DataLoader(
            FlowDataset(
                dataset_dir=dataset_dir,
                splits_dir=splits_dir,
                output_dir=OUTPUT_DIR,
                in_channel=config.c3d_in_channel,
                dataset_percentage=config.dataset_percentage,
                split="val",
                clip_len=16,
            ),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        test_dataloader = DataLoader(
            FlowDataset(
                dataset_dir=dataset_dir,
                splits_dir=splits_dir,
                output_dir=OUTPUT_DIR,
                in_channel=config.c3d_in_channel,
                dataset_percentage=config.dataset_percentage,
                split="test",
                clip_len=16,
            ),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        trainval_loaders = {"train": train_dataloader, "val": val_dataloader}
        trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ["train", "val"]}
        test_size = len(test_dataloader.dataset)

        max_val_acc = 0.0
        for epoch in range(0, config.epochs):
            # each epoch has a training and validation step
            for phase in ["train", "val"]:
                # reset the running loss and corrects
                running_loss = 0.0
                running_corrects = 0.0

                # set model to train() or eval() mode depending on whether it is trained
                # or being validated. Primarily affects layers such as BatchNorm or Dropout.
                if phase == "train":
                    # scheduler.step() is to be called once every epoch during training
                    optimizer.step()
                    model.train()
                else:
                    model.eval()

                for inputs, labels in tqdm(trainval_loaders[phase]):
                    # move inputs and labels to the device the training is taking place on
                    inputs = Variable(inputs.float(), requires_grad=True).to(device)
                    labels = Variable(labels.long()).to(device)
                    optimizer.zero_grad()

                    if phase == "train":
                        outputs = model(inputs)
                    else:
                        with torch.no_grad():
                            outputs = model(inputs)

                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.clip_max_norm
                        )
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    print("running_loss", running_loss)

                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / trainval_sizes[phase]
                epoch_acc = running_corrects.double() / trainval_sizes[phase]

                if phase == "train":
                    wb.log(
                        {
                            "epoch": epoch,
                            "train_loss": epoch_loss,
                            "train_acc": epoch_acc,
                        },
                        step=epoch,
                    )
                else:
                    wb.log(
                        {
                            "epoch": epoch,
                            "val_loss": epoch_loss,
                            "val_acc": epoch_acc,
                        },
                        step=epoch,
                    )

                    if epoch_acc > max_val_acc:
                        print("Found better model.")
                        max_val_acc = epoch_acc
                        torch.save(model.state_dict(), "model.pt")
                        wb.save("model.pt")

                print(
                    "[{}] Epoch: {}/{} Loss: {} Acc: {}".format(
                        phase, epoch + 1, config.epochs, epoch_loss, epoch_acc
                    )
                )

            if useTest and epoch % test_interval == (test_interval - 1):
                model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in tqdm(test_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs = model(inputs)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects.double() / test_size

                wb.log(
                    {
                        "epoch": epoch,
                        "test_loss": epoch_loss,
                        "test_acc": epoch_acc,
                    },
                    step=epoch,
                )

                print(
                    "[test] Epoch: {}/{} Loss: {} Acc: {}".format(
                        epoch + 1, config.epochs, epoch_loss, epoch_acc
                    )
                )


if __name__ == "__main__":
    train_model()

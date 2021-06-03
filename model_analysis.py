"""
This is the result analysis file
1. Confusion matrix
2. Incorrect samples
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from args import get_args
from dataloaders.dataset import RGBDataset
from network import C3D_model, R2Plus1D_model, R3D_model, R3D_BERT
from network.R2Plus1D_BERT import (
    rgb_r2plus1d_16f_34_bert10,
    rgb_r2plus1d_32f_34_bert10,
    rgb_r2plus1d_64f_34_bert10,
)

# For visualizing confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns


args = get_args()
args.skip_frames= True
HMDB_SPLITS_DIR = "./fixtures/hmdb51_splits"
HMDB_RGB_DATASET_DIR = "./data/jpegs_256"
HMDB_FLOW_DATASET_DIR = "./data/tvl1_flow"
# OUTPUT_DIR = f"./data/rgb_output_{str(int(min(args.dataset_percentage, 1) * 100))}"
OUTPUT_DIR = "./data/rgb_output_100"
CLIP_LEN = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataset HMDB51
num_classes = 51
dataset_dir = HMDB_RGB_DATASET_DIR
splits_dir = HMDB_SPLITS_DIR

def eval(model, test_dataloader):
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    y_true, y_pred = [], []
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        tmp = loss.item() * inputs.size(0)
        running_loss += tmp
        print("running_loss", tmp)
        running_corrects += torch.sum(preds == labels.data)
        y_true.extend(list(labels.cpu().numpy()))
        y_pred.extend(list(preds.cpu().numpy()))
    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print(
        "[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc)
    )

    return y_true, y_pred

# Analysis R3D BERT Skip Frames
model_path = "./model/rgb/R3D_BERT_model.pt"
model = R3D_BERT.R3D_BERTClassifier(num_classes=num_classes,
                                in_channels=3,
                                layer_sizes=(2, 2, 2, 2),
                                clip_len=int(CLIP_LEN // 2),
                                pretrained=False)
# Load Pretrained weight
model.load_state_dict(torch.load(model_path))

test_dataloader = DataLoader(
    RGBDataset(
        dataset_dir=dataset_dir,
        splits_dir=splits_dir,
        output_dir=OUTPUT_DIR,
        dataset_percentage=100,
        split="test",
        clip_len=CLIP_LEN,
    ),
    batch_size=40,
    num_workers=0,
)

test_size = len(test_dataloader.dataset)
y_true, y_pred = eval(model, test_dataloader)

conf_matrix = confusion_matrix(y_true, y_pred)
np.save("./plots/conf_maxtrix_R3D_BERT_8", conf_matrix)
conf_matrix = np.load("./plots/conf_maxtrix_R3D_BERT_8.npy")
# Visualize conf_matrix with heatmap
labels2index = test_dataloader.dataset.label2index

class_names = list(labels2index.keys())

def show_confusion_matrix(confusion_matrix, class_names, file_name):
    cm = confusion_matrix.copy()
    # cell_counts = cm.flatten()
    cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    # row_percentages = ['{0:.2f}'.format(value) for value in cm_row_norm.flatten()]

    # cell_labels = [f'{cnt}\n{per}' for cnt, per in zip(cell_counts, row_percentages)]
    # cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])

    df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)
    plt.figure(figsize=(32,24))
    # hmap = sns.heatmap(df_cm, annot=cell_labels, fmt='', cmap='Blues')
    hmap = sns.heatmap(df_cm, fmt='')#, xticklabels=True, yticklabels=True)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=90, ha='right')
    hmap.set(xlabel='True Action', ylabel='Predicted Action', title='R3D_BERT_16_skip_frames')
    # plt.figure(figsize=(64, 48), dpi=80)
    hmap.get_figure().savefig(f"./plots/{file_name}")
    # plt.show()

show_confusion_matrix(conf_matrix, class_names, file_name="R3D_BERT_16_skip_frames_cm.png")


# # Error Case Analysis
# Top Class with best acc and bottom class with lowest acc

acc_per_class = {}
for i in range(num_classes):
    acc_per_class[class_names[i]] = conf_matrix[i][i] / conf_matrix[i].sum()
sorted_keys = sorted(acc_per_class, key=acc_per_class.get)

print("Top 3 Best Class: ", sorted_keys[-3:])
for i in range(1, 4):
    i = -i
    print(f"Accuracy {sorted_keys[i]}: {acc_per_class[sorted_keys[i]]}")


print("Bottom 3 Worest Class: ", sorted_keys[:3])
for i in range(3):
    print(f"Accuracy {sorted_keys[i]}: {acc_per_class[sorted_keys[i]]}")





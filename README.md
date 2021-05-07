# Action Recognition

## Introduction

This repo contains several models for video action recognition,
including C3D, R2Plus1D, R3D, inplemented using PyTorch (0.4.0).
Currently, we train these models on UCF101 and HMDB51 datasets.

## Setup

The code was tested with Anaconda and Python 3.5. After installing the Anaconda environment:

1. Download UCF101 at https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

1. Download pretrained model from [GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).
   Currently, only support pretrained model for C3D.

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    conda install opencv
    pip install tqdm scikit-learn tensorboardX
    ```

1. Change the dataset and pretrained model path in **mypath.py**.

## Train

First time running this command, it will pre-process the video to images.

```shell
python train.py
```

## Datasets

   ```
   UCF-101
   ├── ApplyEyeMakeup
   │   ├── v_ApplyEyeMakeup_g01_c01.avi
   │   └── ...
   ├── ...
   ```

After pre-processing, the output structure is as follows:

   ```
   ucf101
   ├── ApplyEyeMakeup
   │   ├── v_ApplyEyeMakeup_g01_c01
   │   │   ├── 00001.jpg
   │   │   └── ...
   │   └── ...
   ├── ...
   ```

## References

The code is borrowed from or heavily inspired by:

- https://github.com/jfzhang95/pytorch-video-recognition

# Action Recognition

## Setup

The code was tested with Anaconda and Python 3.5. After installing the Anaconda environment:

1. Download [HMDB51](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar), uncompress (including all child rar file inside), move to data/hmdb51_org.

   - data
      - hmdb51_org
         - brush_hair
            - April_09_brush_hair_u_nm_np1_ba_goo_0.avi
            - ...
         - ...

1. (Optional) Download [UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar), uncompress, move to data/UCF-101.

   - data
      - UCF-101
         - ApplyEyeMakeup
            - v_ApplyEyeMakeup_g01_c01.avi
            - ...
         - ...

1. Install dependencies:
   ```shell
   conda install opencv
   pip install tqdm scikit-learn wandb torch torchvision
   ```

1. Change the dataset and pretrained model path in **mypath.py**.

## Train

First time running this command, it will pre-process the video to images.

```shell
make train
```

## References

The code is borrowed from or heavily inspired by:

- https://github.com/jfzhang95/pytorch-video-recognition

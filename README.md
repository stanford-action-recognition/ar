# Action Recognition

## Setup

1. Download HMDB51 dataset

    ```shell
    wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_jpegs_256.zip
    wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_tvl1_flow.zip
    
    mv hmdb51_jpegs_256.zip ar/data
    mv hmdb51_tvl1_flow.zip ar/data
    
    cd ar/data
    
    unzip hmdb51_jpegs_256.zip
    unzip hmdb51_tvl1_flow.zip
    ```

1. Install requriements:
   ```shell
    conda create --name ar python=3.7
    conda activate ar
    conda install opencv
   ```

## Train

### RGB

```shell
python rgb_train.py
```

### Flow

```shell
python motion_train.py
```

### Two Stream

```shell
python multi_stream_train.py --streams=rgb,flow
```

- If use pretrained model and want to freeze the parameters of the pretrained models,  run
  ```shell
  python multi_stream_train.py --streams=rgb,flow --use_pretrained=True --freeze_stream_models=True
  ```
  If use pretrained model without freezing the parameters, note that the new models will replace the pretrained models after training.

- multi_stream_add.py is for summing predictions from the pretrained models of all streams, run
  ```shell
  python multi_stream_add.py --streams=rgb,flow --use_pretrained=True --freeze_stream_models=True
  ```
  This program will print train / val acc.

## References

The code is borrowed from or inspired by:

- https://github.com/artest08/LateTemporalModeling3DCNN
- https://github.com/jfzhang95/pytorch-video-recognition

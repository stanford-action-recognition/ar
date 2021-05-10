### Steps
* Download HMDB51 RGB and Flow datasets from https://github.com/stanford-action-recognition/ar/tree/duanr-dev
* Download hmdb51_splits folder here https://github.com/feichtenhofer/twostreamfusion/tree/master/hmdb51_splits
* Modify global paths specified in rgb_train.py and motion_train.py
  - change HMDB_RGB_DATASET_DIR, HMDB_FLOW_DATASET_DIR, HMDB_SPLITS_DIR to the location of files you downloaded in the previous steps
  - change OUTPUT_DIR to wherever you want
* Run `python rgb_train.py` or `python motion_train.py`

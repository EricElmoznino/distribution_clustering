# Description
Implements (high-dimenstional) clustering algorithm described in https://arxiv.org/pdf/1804.02624.pdf

# Dependencies
```
python3
pytorch (>=0.4)
torchvision
PILLOW
numpy
scipy
tqdm
```


# Usage
First, if you want to use the default `BaseDataset` class, the directory structure of the data you wish to be clustered must conform to the structure shown below.
If another structure makes more sense for your purposes, you will need to sublass the `BaseDataset` class and reference your class in `save_dataset_features.py` and `cluster_dataset.py`.
```
- <data_dir>
    - samples
        - <first file>
        - ...
```

You can then extract deep features from your data by running the command below.
If you are using the `BaseDataset` class, your features will be saved at the path `<data_dir>/features.npy`
```
python sample_dataset_features.py --data_dir <path to data directory>
```

Finally, you can cluster your data by running the command below.
If you are using the `BaseDataset` class, your clustered data will be saved at the path `<data_dir>/clusters`.
Parameters within brackets () are optional.
```
python cluster_dataset.py --data_dir <path to data directory> (--thres <float>) (--min_clus <int>) (--max_dist <float>) (--dont_normalize)
```

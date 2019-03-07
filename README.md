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
First, the directory structure of the data you wish to be clustered must be as follows:
```
- <data_dir>
    - images
        - <first file>
        - ...
```

You can then extract deep features from your data by running the command below.
This will save your data's deep features at the path `<data_dir>/features.npy`
```
python sample_dataset_features.py --data_dir <path to data directory>
```

Finally, you can cluster your data by running the command below.
This will save your clustered data at the path `<data_dir>/clusters`.
Parameters within brackets () are optional.
```
python cluster_dataset.py --data_dir <path to data directory> (--thres <float>) (--min_clus <int>) (--max_dist <float>) (--dont_normalize)
```

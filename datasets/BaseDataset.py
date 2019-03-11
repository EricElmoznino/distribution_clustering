import os
import shutil
import numpy as np


class BaseDataset:

    def __init__(self, data_dir):
        assert os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, 'samples'))
        self.data_dir = data_dir
        self._sample_paths = os.listdir(os.path.join(data_dir, 'samples'))
        self._sample_paths = [s for s in self._sample_paths if s != '.DS_Store']
        self._sample_paths = sorted(self._sample_paths)
        self._sample_paths = [os.path.join(data_dir, 'samples', s) for s in self._sample_paths]
        if os.path.exists(os.path.join(data_dir, 'features.npy')):
            self._features = np.load(os.path.join(data_dir, 'features.npy'))
        else:
            self._features = None

    def sample_paths(self):
        return self._sample_paths

    def features(self):
        assert self._features is not None
        return self._features

    def save_features(self, features):
        features = np.stack(features)
        np.save(os.path.join(self.data_dir, 'features.npy'), features)

    def save_clustered_samples(self, sample_clusters):
        dest = os.path.join(self.data_dir, 'clusters')
        shutil.rmtree(os.path.join(dest), ignore_errors=True)
        os.mkdir(dest)
        for i in range(1, max(sample_clusters) + 1):
            os.mkdir(os.path.join(dest, str(i)))
        for i, sample_path in enumerate(self._sample_paths):
            if sample_clusters[i] == 0:
                continue
            clus = str(sample_clusters[i])
            shutil.copyfile(sample_path, os.path.join(dest, clus, sample_path.split('/')[-1]))

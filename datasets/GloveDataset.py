import os
import numpy as np

from datasets.BaseDataset import BaseDataset


class GloveDataset(BaseDataset):

    def __init__(self, data_dir):
        assert os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, 'word_list.txt')) and \
               os.path.exists(os.path.join(data_dir, 'word_vectors.npy'))
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'word_list.txt')) as f:
            self.words = f.read().split('\n')
        self._features = np.load(os.path.join(data_dir, 'word_vectors.npy'))

    def sample_paths(self):
        raise NotImplementedError('This dataset assumes features already exist, '
                                  'so itterating sample paths is not necessary '
                                  '(i.e. do not run save_dataset_features.py).')

    def save_features(self, features):
        raise NotImplementedError('This dataset assumes features already exist, '
                                  'so they should not be saved '
                                  '(i.e. do not run save_dataset_features.py).')

    def save_clustered_samples(self, sample_clusters):
        dest = os.path.join(self.data_dir, 'clusters.txt')
        clustered_words = [[] for _ in range(max(sample_clusters))]
        for i, word in enumerate(self.words):
            if sample_clusters[i] == 0:
                continue
            clustered_words[sample_clusters[i]].append(word)
        with open(dest, 'w') as f:
            for i, cluster in enumerate(clustered_words):
                f.write('%d:\n' % i)
                for word in cluster:
                    f.write('\t%s\n' % word)

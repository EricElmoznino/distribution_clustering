import unittest
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as tr
from PIL import Image
import shutil
import os

from distribution_clustering import cluster, cluster_with_model_features


class ColiseumTests(unittest.TestCase):

    def test_cluster(self):
        dir = 'unit_tests/data/coliseum'

        features = np.load(os.path.join(dir, 'matlab_features.npy'))
        _, sample_clusters, cluster_distances = cluster(features, normalize=False)

        expected_cluster_distances = {1: 1.0324226759046613, 2: 1.1990520335796435, 3: 1.2652137201245968,
                                      4: 1.5818832300458632, 5: 1.6069637516834046, 6: 1.6155547275177278,
                                      7: 1.7198951098517856, 8: 1.723933851763933, 9: 1.7777007531041384,
                                      10: 1.7851095182277106, 11: 1.818438112519426, 12: 1.8839663342425441}
        for c in expected_cluster_distances:
            assert c in cluster_distances
            assert cluster_distances[c] == expected_cluster_distances[c]

        images = ['img' + str(i).zfill(6) + '.jpg' for i in range(100)]
        images = [os.path.join(dir, 'samples', img) for img in images]
        save_cluster_images(sample_clusters,
                            images=images,
                            dest=os.path.join(dir, 'clustered_from_features'))

    def test_cluster_with_model_features(self):

        class TestModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.features = models.alexnet(pretrained=True).features

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                return x

        class TestDataset(Dataset):

            def __init__(self, data_dir):
                super().__init__()
                self.data = os.listdir(data_dir)
                self.data = [d for d in self.data if '.jpg' in d]
                self.data = [os.path.join(data_dir, d) for d in self.data]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, item):
                image = self.data[item]
                image = Image.open(image).convert('RGB')
                image = tr.resize(image, 224)
                image = tr.to_tensor(image)
                image = tr.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                return image

        dir = 'unit_tests/data/coliseum'

        model = TestModel()
        model.eval()
        dataloader = DataLoader(TestDataset(os.path.join(dir, 'samples')))
        _, sample_clusters, _ = cluster_with_model_features(model, dataloader)

        save_cluster_images(sample_clusters,
                            images=dataloader.dataset.data,
                            dest=os.path.join(dir, 'clustered_from_model_features'))


def save_cluster_images(sample_clusters, images, dest):
    shutil.rmtree(os.path.join(dest), ignore_errors=True)
    os.mkdir(dest)
    for i in range(1, max(sample_clusters) + 1):
        os.mkdir(os.path.join(dest, str(i)))
    for i, image in enumerate(images):
        if sample_clusters[i] == 0:
            continue
        clus = str(sample_clusters[i])
        shutil.copyfile(image,
                        os.path.join(dest, clus, image.split('/')[-1]))

import os
import shutil
import numpy as np
import time
import argparse

from distribution_clustering import cluster


def save_cluster_images(sample_clusters, images, dest):
    shutil.rmtree(os.path.join(dest), ignore_errors=True)
    os.mkdir(dest)
    for i in range(1, max(sample_clusters) + 1):
        os.mkdir(os.path.join(dest, str(i)))
    for i, image in enumerate(images):
        if sample_clusters[i] == 0:
            continue
        clus = str(sample_clusters[i])
        shutil.copyfile(image, os.path.join(dest, clus, image.split('/')[-1]))


# Parse arguments
parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--data_dir', default=None, help='directory of data to be clustered.')
cluster_grp = parser.add_argument_group('clustering parameters')
cluster_grp.add_argument('--thres', default=0.07, type=float,
                         help='threshold for clustering.')
cluster_grp.add_argument('--min_clus', default=5, type=int,
                         help='minimum number of clusters.')
cluster_grp.add_argument('--max_dist', default=2.0, type=float,
                         help='maximum distance for clustering.')
cluster_grp.add_argument('--dont_normalize', dest='normalize', action='store_false',
                         help='don\'t normalize features before clustering.')
parser.set_defaults(normalize=True)
args = parser.parse_args()

# Load features and do clustering
images = os.listdir(os.path.join(args.data_dir, 'images'))
images = [img for img in images if img != '.DS_Store']
images = sorted(images)
images = [os.path.join(args.data_dir, 'images', img) for img in images]
features = np.load(os.path.join(args.data_dir, 'features.npy'))
start_time = time.time()
_, sample_clusters, cluster_distances = cluster(features, thres=args.thres, min_clus=args.min_clus,
                                                max_dist=args.max_dist, normalize=args.normalize)
end_time = time.time()

# Save images into their corresponding clusters
save_cluster_images(sample_clusters, images=images, dest=os.path.join(args.data_dir, 'clusters'))

# Print results
print('Complete')
print('Found %d clusters' % max(sample_clusters))
print('Clustered %d images out of %d' % (len([c for c in sample_clusters if c > 0]), len(sample_clusters)))
total_seconds = end_time - start_time
hours = int(total_seconds // (60 * 60))
mins = int((total_seconds // 60) % 60)
print('Total clustering time: %d hours, %d minutes' % (hours, mins))

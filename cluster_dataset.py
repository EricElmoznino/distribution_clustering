import time
import argparse

from distribution_clustering import cluster
from datasets.BaseDataset import BaseDataset


# Parse arguments
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--data_dir', default=None, help='directory of data to be clustered.')
cluster_grp = parser.add_argument_group('clustering parameters')
cluster_grp.add_argument('--thres', default=0.07, type=float,
                         help='second order distance for acceptance to a cluster.')
cluster_grp.add_argument('--min_clus', default=5, type=int,
                         help='minimum number of samples for creation of a clutser.')
cluster_grp.add_argument('--max_dist', default=2.0, type=float, choices=[Range(0.0, 2.0)],
                         help='ignore higher variance classes for speed, maximum is 2.')
cluster_grp.add_argument('--dont_normalize', dest='normalize', action='store_false',
                         help='don\'t normalize features before clustering.')
parser.set_defaults(normalize=True)
args = parser.parse_args()

# Load features and do clustering
dataset = BaseDataset(args.data_dir)
start_time = time.time()
_, sample_clusters, cluster_distances = cluster(dataset.features(), thres=args.thres, min_clus=args.min_clus,
                                                max_dist=args.max_dist, normalize=args.normalize)
end_time = time.time()

# Save samples into their corresponding clusters
dataset.save_clustered_samples(sample_clusters)

# Print results
print('Complete')
print('Found %d clusters' % max(sample_clusters))
print('Clustered %d samples out of %d' % (len([c for c in sample_clusters if c > 0]), len(sample_clusters)))
total_seconds = end_time - start_time
hours = int(total_seconds // (60 * 60))
mins = int((total_seconds // 60) % 60)
print('Total clustering time: %d hours, %d minutes' % (hours, mins))

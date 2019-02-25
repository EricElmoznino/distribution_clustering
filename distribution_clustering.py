import numpy as np
import scipy.spatial
import torch


def cluster(features, thres=0.07, min_clus=5, max_dist=2.0, normalize=True):
    assert len(features) > 0 and len(features.shape) == 2

    if normalize:
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norms[feat_norms == 0] = 1
        features /= feat_norms

    pair_dist = scipy.spatial.distance.pdist(features, 'sqeuclidean')
    pair_dist = scipy.spatial.distance.squareform(pair_dist)

    # Loop initialization
    inf = 1000.0
    pair_dist_base = pair_dist.copy()
    pair_dist = pair_dist + inf * np.identity(len(features))
    sample_clusters = np.zeros(len(features), dtype=np.int)
    cluster_distances = {}
    cur_cluster = 1
    finished = False

    while not finished:
        finished = True

        if (sample_clusters > 0).sum() < len(features):
            i, j = np.unravel_index(pair_dist.argmin(), pair_dist.shape)
            cur_dist = pair_dist[i, j]
            pair_dist[i, j] = pair_dist[j, i] = inf
            cur_vec = compute_vec(pair_dist_base, [i, j], cur_dist)

            a, b = pair_dist_base[i, :].copy(), pair_dist_base[j, :].copy()
            a[j] = b[i] = 0

            if np.abs(a - b).mean() > thres and cur_dist <= max_dist:
                finished = False
                continue

            if cur_dist == 0:
                finished = False
                continue

            if cur_dist <= max_dist:
                clus = sample_clusters.copy()
                clus[i] = clus[j] = cur_cluster
                clus = clus_strange(pair_dist_base, clus, cur_vec, cur_dist, thres, thres, cur_cluster)

                loc_ind = np.argwhere(clus == cur_cluster).flatten()

                cur_dist = compute_dist(pair_dist_base, loc_ind)
                cur_vec = compute_vec(pair_dist_base, loc_ind, cur_dist)
                clus = clus_strange(pair_dist_base, clus, cur_vec, cur_dist, thres, thres, cur_cluster)

                if (clus == cur_cluster).sum() > min_clus:
                    sample_clusters = clus
                    cluster_distances[cur_cluster] = cur_dist
                    cur_cluster += 1
                    pair_dist[sample_clusters > 0, :] = pair_dist[:, sample_clusters > 0] = inf

                finished = False

        if cur_dist > max_dist:
            finished = True

    if normalize:
        features *= feat_norms

    centers = [features[sample_clusters == c].mean(axis=0)
               for c in range(1, max(sample_clusters) + 1)]

    return centers, sample_clusters.tolist(), cluster_distances


def cluster_with_model_features(model, dataloader, thres=0.07, min_clus=5, max_dist=2.0, normalize=True):
    features = None
    with torch.no_grad():
        for input in dataloader:
            output = model(input).cpu().numpy()
            if features is None:
                features = output
            else:
                features = np.concatenate((features, output))
    return cluster(features, thres=thres, min_clus=min_clus, max_dist=max_dist, normalize=normalize)


def compute_vec(d, indices, dist):
    d = d[indices, :].copy()
    for i in range(len(indices)):
        d[i, indices[i]] = dist
    return d.mean(axis=0)


def compute_dist(d, indices):
    d = d[indices, :][:, indices]
    s = len(d)
    return d.sum() / (s ** 2 - s)


def clus_strange(d, clus, cur_vec, cur_dist, thres_all, thres_loc, cur_clus):
    change = True
    while change:
        change = False
        for i in range(len(clus)):
            if clus[i] != cur_clus:
                continue
            for j in range(len(clus)):
                if i == j or clus[j] > 0:
                    continue
                vec_j = d[j, :].copy()
                vec_j[j] = cur_dist
                dist = np.abs(cur_vec - vec_j).mean()
                t = d[j, :][clus == cur_clus].mean()
                if dist < thres_all and abs(t - cur_dist) < thres_loc:
                    clus[j] = clus[i]
                    change = True
    return clus

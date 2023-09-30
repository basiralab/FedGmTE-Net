import torch
import numpy as np
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist

# Vectorize given matrix (Based on Lower Triangular - Could use upper as well)
def vectorize(M):
    return M[np.tril_indices(M.shape[0], k=-1)]

# Antivectorize given vector
def antiVectorize(vec, m):
    M = np.zeros((m,m))
    M[np.tril_indices(m,k=-1)] = vec
    M= M.transpose()
    M[np.tril_indices(m,k=-1)] = vec
    return M

# Antivectorize given vector for torch
def antiVectorize_tensor(vec, m, device):
    M = torch.zeros((m, m)).to(device)
    idx = torch.tril_indices(m, m, offset=-1).to(device)
    M[idx[0], idx[1]] = vec
    M = M.transpose(0, 1)
    M[idx[0], idx[1]] = vec
    return M

# CV splits
def get_nfold_split(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds, shuffle=False)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    return X_train, X_test

def kmeans(X, k, max_iter=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Assign each point to its nearest centroid
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Check if any cluster has less than 2 samples
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        if np.any(label_counts < 2):
            # Find clusters with less than 2 samples
            clusters_less_than_2 = unique_labels[label_counts < 2]

            # Randomly assign additional samples to those clusters
            for cluster in clusters_less_than_2:
                candidates = np.where(labels != cluster)[0]
                additional_sample = np.random.choice(candidates, size=2, replace=False)
                labels[additional_sample] = cluster

        # Update centroids based on the mean of the points assigned to them
        for j in range(k):
            centroids[j] = np.mean(X[labels == j], axis=0)

    return labels

def construct_id_adjacency_matrix(features):
    adjacency_matrix = torch.eye(len(features))

    return adjacency_matrix

def construct_similarity_adjacency_matrix(features):
    n_nodes = features.shape[0]
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            similarity = np.exp(-np.linalg.norm(features[i].cpu() - features[j].cpu()))
            adjacency_matrix[i, j] = similarity
            adjacency_matrix[j, i] = similarity

    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)

    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

    return adjacency_matrix

def normalize_adjacency_matrix(adj):
    row_sums = np.sum(adj, axis=1)
    D = np.diag(row_sums)
    D_inv = np.linalg.inv(D)

    adj_norm = D_inv.dot(adj)

    return adj_norm

def random_table(num_samples, num_time, ratio=1/2):
    """
        Returns a 2D table where each slot is randomly filled with zero or one based on a ratio
        samples x timepoints
        at t0 always 1
    """

    table = np.ones((num_samples, num_time))
    comb = np.zeros((num_samples) * (num_time - 1))
    comb[: int(ratio * comb.shape[0])] = 1
    np.random.shuffle(comb)
    comb = comb.reshape(num_samples, num_time - 1)
    table[:, 1:] = comb

    return table

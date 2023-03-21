# Tools to do feature selection 

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def laplacian_score(X, k=5, t=None):
    """
    Calculate Laplacian Score for each feature in the dataset.
    
    Parameters:
    X: (n_samples, n_features) - Input data
    k: int - Number of nearest neighbors to consider
    t: float - Heat kernel parameter (optional)
    
    Returns:
    scores: (n_features,) - Laplacian Scores for each feature
    """
    n_samples, n_features = X.shape

    # Compute the pairwise Euclidean distance matrix
    dist_matrix = squareform(pdist(X))

    # Calculate the k-nearest neighbors adjacency matrix
    W = kneighbors_graph(X, k, mode='connectivity', include_self=True).toarray()

    # Compute the heat kernel (optional)
    if t is not None:
        W = np.exp(-dist_matrix ** 2 / t)

    # Normalize the adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    S = np.dot(D_inv_sqrt, np.dot(W, D_inv_sqrt))

    # Compute the Laplacian matrix
    L = csgraph.laplacian(W, normed=True)

    # Calculate Laplacian Score for each feature
    scores = []
    for i in range(n_features):
        fi = X[X.columns[i]]
        fi = fi - np.dot(S, fi).sum() / n_samples
        L_score = np.dot(fi.T, np.dot(L, fi)) / np.dot(fi.T, np.dot(D, fi))
        scores.append(L_score)

    return np.array(scores)



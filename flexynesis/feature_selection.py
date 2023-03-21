# Tools to do feature selection 

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def laplacian_score(X, k=5, t=None):
    """
    Calculate Laplacian Score for each feature in the dataset.
    
    Parameters:
    X: numpy array (n_samples, n_features) - Input data
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
        fi = X[:,i]
        fi = fi - np.dot(S, fi).sum() / n_samples
        L_score = np.dot(fi.T, np.dot(L, fi)) / np.dot(fi.T, np.dot(D, fi))
        scores.append(L_score)

    return np.array(scores)

def remove_redundant_features(X, laplacian_scores, threshold=0.8):
    """
    Efficiently selects features based on Laplacian scores while avoiding highly correlated features.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        The input data matrix with samples in rows and features in columns.
    laplacian_scores : numpy array, shape (n_features,)
        The Laplacian scores for each feature in the input data matrix X.
    threshold : float, optional, default: 0.8
        The Pearson correlation coefficient threshold for removing highly correlated features.
        Features with correlation coefficients greater than the threshold will be removed.

    Returns
    -------
    selected_features : list of int
        The list of selected feature indices in the input data matrix X.
    """

    # Compute the correlation matrix once
    correlation_matrix = np.corrcoef(X, rowvar=False)
    
    # Rank the features by Laplacian scores
    ranked_indices = np.argsort(laplacian_scores)

    selected_features = []

    for idx in ranked_indices:
        correlated = False

        for selected_idx in selected_features:
            # Check the absolute correlation between the features
            correlation = np.abs(correlation_matrix[idx, selected_idx])

            if correlation > threshold:
                correlated = True
                break

        if not correlated:
            selected_features.append(idx)

    return selected_features

def filter_by_laplacian(X, k=5, t=None, topN = 100, remove_redundant = True, threshold = 0.8):
    """
    Given a data matrix, compute laplacian score for each feature
    and return a filtered data matrix based on top laplacian scores.
    
    Parameters:
    X: Pandas Data Frame (n_samples, n_features) - Input data
    k: int - Number of nearest neighbors to consider
    t: float - Heat kernel parameter (optional)
    topN = Top number of features to keep
    remove_redundant = (Boolean) whether to remove highly correlated 
        features based on laplacian score ranking. Default: True. 
    Returns:
    X_selected: (n_samples, n_features) - Filtered data matrix
    """
    # compute laplacian scores
    scores = laplacian_score(X.values, k, t)
    
    # Sort the features based on their Laplacian Scores
    sorted_indices = np.argsort(scores)

    if remove_redundant:
        # sorted indices filtered for correlation
        sorted_indices = remove_redundant_features(X.values, scores, threshold)
        
    # Choose the top k features with the lowest Laplacian Scores
    selected_features = sorted_indices[:topN]

    # Extract the selected features from the dataset
    X_selected = X[X.columns[selected_features]]
    
    return X_selected
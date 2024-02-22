# Tools to do feature selection 

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

from scipy.sparse import csr_matrix, diags
from tqdm import tqdm

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
    W = kneighbors_graph(X, k, mode='connectivity', include_self=True)

    # Compute the heat kernel (optional)
    if t is not None:
        W = csr_matrix(np.exp(-dist_matrix ** 2 / t))

    # Normalize the adjacency matrix
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(D))

    S = D_inv_sqrt @ W @ D_inv_sqrt
    S = S.toarray()

    # Compute the Laplacian matrix
    L = csgraph.laplacian(W, normed=True)

    # Calculate Laplacian Score for each feature
    scores = []
    D = diags(D)  # Convert to sparse diagonal matrix for efficient dot product
    for i in tqdm(range(n_features), desc='Calculating Laplacian scores'):
        fi = X[:,i]
        fi = fi - np.dot(S, fi).sum() / n_samples
        L_score = np.dot(fi.T, L @ fi) / np.dot(fi.T, D @ fi)
        scores.append(L_score)

    return np.array(scores)


def remove_redundant_features(X, laplacian_scores, threshold, topN=None):
    """
    Selects features based on Laplacian scores while avoiding highly correlated features. 

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        Input data matrix with samples in rows and features in columns.
    laplacian_scores : numpy array, shape (n_features,)
        Laplacian scores for each feature in X.
    threshold : float, optional (default=0.8)
        Pearson correlation coefficient threshold for removing correlated features.
    topN : int, optional (default=None)
        Number of features to return, up to this number, prioritizing non-redundant ones.

    Returns
    -------
    selected_features : list of int
        Indices of selected features in X.
    """
    
    # Precompute the correlation matrix once
    correlation_matrix = np.corrcoef(X, rowvar=False)

    # Rank the features by Laplacian scores in descending order (from highest to lowest)
    ranked_indices = np.argsort(laplacian_scores)[::-1]

    selected_features = []
    redundant_features = []

    for idx in tqdm(ranked_indices, desc='Filtering redundant features'):
        correlated = False

        for selected_idx in selected_features:
            # Check the absolute correlation between the features
            correlation = np.abs(correlation_matrix[idx, selected_idx])

            if correlation > threshold:
                correlated = True
                break

        if correlated:
            redundant_features.append(idx)
        else:
            selected_features.append(idx)

    # If fewer than topN features are selected, top up from the redundant list
    if topN and len(selected_features) < topN:
        shortfall = topN - len(selected_features)
        selected_features.extend(redundant_features[:shortfall])

    return selected_features


def filter_by_laplacian(X, layer, k=5, t=None, topN=100, correlation_threshold=0.9):
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
    
    print("Implementing feature selection using laplacian score for layer:",layer,"with ",X.shape[1],"features")
    
    # only apply filtering if topN < n_features
    if topN >= X.shape[1]: 
        print("Returning original matrix, demanded # of features is ", 
        "larger than existing number of features")
        return X
    
    # compute laplacian scores
    scores = laplacian_score(X.values, k, t)
    
    # Sort the features based on their Laplacian Scores
    sorted_indices = np.argsort(scores)
    selected_features = sorted_indices[:topN]

    if correlation_threshold < 1:
        # Choose the topN + 10% features with the lowest Laplacian Scores
        # this is done to avoid unnecessary computation of correlation for all features. 
        topN_extended = int(topN + 0.10 * X.shape[1])
        topN_extended = min(topN_extended, X.shape[1])  # Ensure we don't exceed the number of features
        selected_features = sorted_indices[:topN_extended]

        # Remove redundancy from topN + 10% features
        selected_features = remove_redundant_features(X[X.columns[selected_features]].values, 
                                                      scores[selected_features], correlation_threshold, 
                                                      topN)
        # Prune down to topN features
        selected_features = selected_features[:topN]

    # Extract the selected features from the dataset
    X_selected = X[X.columns[selected_features]]
    
    return X_selected

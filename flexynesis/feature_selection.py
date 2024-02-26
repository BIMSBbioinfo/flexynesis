# Tools to do feature selection 

import numpy as np
import pandas as pd
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
    Removes features from the dataset based on correlation and Laplacian scores, optionally 
    keeping only the top N features based on their scores, and manages redundant features.

    This function evaluates features in a dataset for redundancy by measuring the correlation 
    between them. If the absolute correlation between two features exceeds a specified threshold, 
    the feature with the higher Laplacian score (a lower score is better) is considered redundant 
    and is marked for removal. 
    If the number of selected features is less than a specified top N, additional features are 
    included from the redundant set based on their Laplacian scores until the top N count is reached.

    Parameters
    ----------
    X : pandas DataFrame
        The input dataset with samples in rows and features in columns.
    laplacian_scores : numpy array
        An array of Laplacian scores corresponding to each feature in the dataset. Lower scores
        indicate features that better capture the intrinsic geometry of the data.
    threshold : float
        The correlation threshold above which a feature is considered redundant. Features with
        absolute correlation greater than this threshold with any selected feature are marked
        as redundant.
    topN : int, optional
        The desired number of features to select. If specified, the function will ensure that
        exactly topN features are selected, including from the redundant set if necessary.
        If None, no limit is applied to the number of selected features.

    Returns
    -------
    selected_features : list
        A list of indices of the features selected to be retained in the dataset.
    redundant_features_df : pandas DataFrame
        A DataFrame listing the features considered redundant, the indices of the features
        they are most correlated with, and the correlation scores.

    Notes
    -----
    - The function prioritizes the removal of features with higher Laplacian scores when
      deciding between two correlated features, under the assumption that lower scores
      indicate features more important for preserving data structure.
    - Features initially marked as redundant but included during the top-up process to meet
      the topN requirement are removed from the redundant_features_df before it is returned.
    """    
    correlation_matrix = np.corrcoef(X.T)
    ranked_indices = np.argsort(laplacian_scores)  # Assuming minimizing laplacian_scores
    selected_features = []
    redundant_features = {}

    for idx in tqdm(ranked_indices, desc='Filtering redundant features'):
        correlated = False
        for selected_idx in selected_features:
            correlation = np.abs(correlation_matrix[idx, selected_idx])
            if correlation > threshold:
                correlated = True
                correlated_feature = selected_idx
                correlation_score = correlation
                break

        if correlated:
            redundant_features[idx] = {'correlated_with': correlated_feature, 'correlation_score': correlation_score}
        else:
            selected_features.append(idx)

    # Topping up from redundant features if fewer than topN features are selected
    if topN is not None and len(selected_features) < topN:
        shortfall = topN - len(selected_features)
        # Sort redundant features by their laplacian score, prioritizing lower scores
        sorted_redundant_indices = sorted(redundant_features.keys(), key=lambda x: laplacian_scores[x])
        topped_up_features = []
        for idx in sorted_redundant_indices:
            if len(selected_features) < topN:
                selected_features.append(idx)
                topped_up_features.append(idx)
            else:
                break  # Stop if we have reached the desired number of features

        # Remove topped-up features from the redundant_features dictionary
        for idx in topped_up_features:
            del redundant_features[idx]
    
    if len(redundant_features) > 0:
        # Convert redundant_features dictionary to DataFrame
        redundant_features_df = pd.DataFrame([
            {
                "feature": X.columns[idx],
                "correlated_with": X.columns[redundant_features[idx]['correlated_with']],
                "correlation_score": redundant_features[idx]['correlation_score']
            }
            for idx in redundant_features
        ])
        return X.columns[selected_features], redundant_features_df
    else:
        return X.columns[selected_features], pd.DataFrame()

def filter_by_laplacian(X, layer, k=5, t=None, topN=100, correlation_threshold=0.9):
    """
    Filters features in a dataset based on Laplacian scores and removes highly correlated features, 
    retaining only the top N features with the lowest scores and optionally considering correlation.

    This function computes Laplacian scores for each feature in the dataset to measure its importance
    in capturing the data's intrinsic geometry. It then selects the top N features with the lowest scores.
    Additionally, it can remove features that are highly correlated beyond a specified threshold to reduce
    redundancy. The selection process can extend to include slightly more features before correlation filtering
    to ensure the best candidates are retained.

    Parameters
    ----------
    X : pandas.DataFrame
        The input dataset with samples in rows and features in columns.
    layer : str
        Identifier for the dataset layer being processed (for logging purposes).
    k : int, optional
        The number of nearest neighbors to consider for computing the Laplacian score (default is 5).
    t : float, optional
        The heat kernel parameter for Laplacian score computation. If None, the default behavior 
        applies without a heat kernel (default is None).
    topN : int, optional
        The number of top features to keep based on the lowest Laplacian scores (default is 100).
    correlation_threshold : float, optional
        The Pearson correlation coefficient threshold for identifying and removing highly correlated 
        features. Features with a correlation above this threshold are considered redundant 
        (default is 0.9).

    Returns
    -------
    X_selected : pandas.DataFrame
        The filtered data matrix with only the selected features retained.
    feature_log : pandas.DataFrame
        A DataFrame logging each feature's Laplacian score, correlation information (if applicable),
        and whether it was selected. It includes columns for 'feature', 'laplacian_score',
        'correlated_with' (feature causing redundancy), 'correlation_score', and 'selected'
        (a boolean indicating if the feature is retained in the final selection).

    Notes
    -----
    - The function prioritizes features with lower Laplacian scores as they are considered
      more important for capturing the data's structure.
    - Features are initially selected based on their Laplacian scores. If `correlation_threshold`
      is set below 1, the function further filters these features by removing those that are highly
      correlated with any already selected feature.
    - The process may select additional features beyond `topN` before correlation filtering to ensure
      that the best candidates are considered. The final number of features, however, is pruned to `topN`.
    """    
    print("[INFO] Implementing feature selection using laplacian score for layer:",layer,"with ",X.shape[1],"features")
    
    feature_log = pd.DataFrame({'feature': X.columns, 'laplacian_score': np.nan})
    # only apply filtering if topN < n_features
    if topN >= X.shape[1]: 
        print("[INFO] No feature selection applied. Returning original matrix. Demanded # of features is ", 
        "larger than existing number of features")
        return X, feature_log
    
    # compute laplacian scores
    scores = laplacian_score(X.values, k, t)
    
    feature_log = pd.DataFrame({'feature': X.columns, 'laplacian_score': scores})
    
    # Sort the features based on their Laplacian Scores
    sorted_indices = np.argsort(scores)
    selected_feature_indices = sorted_indices[:topN]
    selected_features = X.columns[selected_feature_indices]

    if correlation_threshold < 1:
        # Choose the topN + 10% features with the lowest Laplacian Scores
        # this is done to avoid unnecessary computation of correlation for all features. 
        topN_extended = int(topN + 0.10 * X.shape[1])
        topN_extended = min(topN_extended, X.shape[1])  # Ensure we don't exceed the number of features
        selected_features = sorted_indices[:topN_extended]

        # Remove redundancy from topN + 10% features
        selected_features, redundant_features_df = remove_redundant_features(X[X.columns[selected_feature_indices]], 
                                                      scores[selected_feature_indices], correlation_threshold, 
                                                      topN)
        # Prune down to topN features
        selected_features = selected_features[:topN]
        
        # if any redundant features found, merge feature log with info from this. 
        if not redundant_features_df.empty:
            # record the table of features which were removed due to redundancy
            feature_log = pd.merge(feature_log, redundant_features_df, on = 'feature', how = 'outer')

    # Extract the selected features from the dataset
    X_selected = X[selected_features]
    
    feature_log['selected'] = False
    feature_log.loc[feature_log['feature'].isin(selected_features),'selected'] = True
    
    return X_selected, feature_log

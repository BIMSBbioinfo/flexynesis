# functions to create distorted versions of given data matrices 

import numpy as np
from sklearn.decomposition import PCA

def distort_pca(matrix, pc_index, jitter_factor):
    """
    Applies PCA on the input matrix, distorts the selected principal component, 
    and reconstructs the matrix using the distorted principal components.
    
    Args:
    matrix (np.array): Input data matrix (n_samples, n_features)
    pc_index (int): Index of the principal component to distort (0-indexed)
    jitter_factor (float): Factor by which to distort the selected principal component
    
    Returns:
    np.array: Reconstructed matrix with distorted principal component
    """
    # Compute PCA
    pca = PCA()
    transformed_matrix = pca.fit_transform(matrix)

    # Distort selected principal component with jitter factor
    transformed_matrix[:, pc_index] *= jitter_factor

    # Re-compute the original data matrix using the distorted principal components
    reconstructed_matrix = pca.inverse_transform(transformed_matrix)

    return reconstructed_matrix


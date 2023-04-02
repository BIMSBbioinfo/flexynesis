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

def distort_multiomic_dataset(dataset, pc_indices, jitter_factors):
    """
    Create a new MultiomicDataset with distorted versions of each matrix using the distort_pca function.
    Distort the principal components sequentially using the corresponding jitter factors.

    Args:
    dataset (MultiomicDataset): An instance of the MultiomicDataset class
    pc_indices (list): List of indices of the principal components to distort (0-indexed)
    jitter_factors (list): List of factors by which to distort the selected principal components

    Returns:
    MultiomicDataset: A new MultiomicDataset with distorted matrices
    """
    if len(pc_indices) != len(jitter_factors):
        raise ValueError("The lengths of pc_indices and jitter_factors must be equal")

    distorted_dat = {}
    for data_type, tensor in dataset.dat.items():
        distorted_tensor = tensor.clone()
        numpy_matrix = distorted_tensor.numpy()
        for pc_index, jitter_factor in zip(pc_indices, jitter_factors):
            numpy_matrix = distort_pca(numpy_matrix, pc_index, jitter_factor)
        distorted_tensor = torch.tensor(numpy_matrix, dtype=tensor.dtype, device=tensor.device)
        distorted_dat[data_type] = distorted_tensor

    return MultiomicDataset(distorted_dat, dataset.y, dataset.features, dataset.samples)

def concatenate_datasets(dataset1, dataset2):
    """
    Concatenate two MultiomicDataset objects.

    Args:
    dataset1 (MultiomicDataset): The first MultiomicDataset object
    dataset2 (MultiomicDataset): The second MultiomicDataset object

    Returns:
    MultiomicDataset: A new MultiomicDataset with concatenated matrices and labels
    """
    # Check if both datasets have the same data types
    if set(dataset1.dat.keys()) != set(dataset2.dat.keys()):
        raise ValueError("Both datasets must have the same data types")

    concatenated_dat = {}
    for data_type, matrix1 in dataset1.dat.items():
        matrix2 = dataset2.dat[data_type]
        concatenated_matrix = np.concatenate((matrix1, matrix2), axis=0)
        concatenated_dat[data_type] = concatenated_matrix

    concatenated_y = torch.cat((dataset1.y, dataset2.y), axis = 0)
    concatenated_samples = dataset1.samples + dataset2.samples

    return MultiomicDataset(concatenated_dat, concatenated_y, dataset1.features, concatenated_samples)


def augment_dataset_with_pc_distortion(dataset, pc_indices, interval, n):
    """
    Create a list of size N of distorted MultiomicDataset objects, append to the initial dataset 

    Args:
    dataset (MultiomicDataset): An instance of the MultiomicDataset class
    pc_indices (list): List of indices of the principal components to distort (0-indexed)
    interval (tuple): A tuple defining the allowed interval for the jitter factor (e.g., (0.5, 1.5))
    n (int): The number of distorted datasets to create

    Returns:
    N+1 MultiomicDatasets concatenated
    """
    distorted_datasets = [dataset]
    for _ in range(n):
        jitter_factors = [random.uniform(interval[0], interval[1]) for _ in pc_indices]
        distorted_dataset = distort_multiomic_dataset(dataset, pc_indices, jitter_factors)
        distorted_datasets.append(distorted_dataset)
    
    # merge datasets
    return reduce(concatenate_datasets, distorted_datasets)

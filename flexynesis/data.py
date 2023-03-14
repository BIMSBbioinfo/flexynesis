from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import reduce
import torch

class MultiomicDataset(Dataset):
    """A PyTorch dataset for multiomic data.

    Args:
        dat (dict): A dictionary with keys corresponding to different types of data and values corresponding to matrices of the same shape. All matrices must have the same number of samples (rows).
        y (list or np.array): A 1D array of labels with length equal to the number of samples.
        features (list or np.array): A 1D array of feature names with length equal to the number of columns in each matrix.
        samples (list or np.array): A 1D array of sample names with length equal to the number of rows in each matrix.

    Returns:
        A PyTorch dataset that can be used for training or evaluation.
    """

    def __init__(self, dat, y, features, samples):
        """Initialize the dataset."""
        self.dat = dat
        self.y = y
        self.features = features
        self.samples = samples 

    def __getitem__(self, index):
        """Get a single data sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A tuple of two elements: 
                1. A dictionary with keys corresponding to the different types of data in the input dictionary `dat`, and values corresponding to the data for the given sample.
                2. The label for the given sample.
        """
        return {x: self.dat[x][index] for x in self.dat.keys()}, self.y[index]

    def __len__ (self):
        """Get the total number of samples in the dataset.

        Returns:
            An integer representing the number of samples in the dataset.
        """
        return len(self.y)

    
def get_labels(dat, drugs, drugName, batch_size):
    y = drugs[drugName]
    y = y[~y.isna()]

    # list of samples in the assays
    samples = list(reduce(set.intersection, [set(item) for item in [dat[x].columns for x in dat.keys()]]))
    # keep samples with labels
    samples = list(set(y.index).intersection(samples))
    if len(samples) % batch_size == 1:
        # I do this to avoid batches of size 1
        samples = samples[0:len(samples)-1]
    #subset assays and labels for the remaining samples
    dat = {x: dat[x][samples] for x in dat.keys()}
    y = y[samples]
    return dat, y

# dat: list of matrices, features on the rows, samples on the columns
# ann: pandas data frame with 'y' as variable for the corresponding samples on the matrix columns
# task_type: type of outcome (classification/regression)
# notice the tables are transposed to return samples on rows, features on columns
def get_torch_dataset(dat, labels):
    # keep a copy of row/column names
    features = {x: dat[x].index for x in dat.keys()}
    samples = {x: dat[x].columns for x in dat.keys()}
    dat = {x: torch.from_numpy(np.array(dat[x].T)).float() for x in dat.keys()}
    y =  torch.from_numpy(np.array(labels)).float()
    return MultiomicDataset(dat, y, features, samples)

def make_dataset(dat, *args):
    dat, y = get_labels(dat, *args)
    dataset = get_torch_dataset(dat, y)
    return dataset
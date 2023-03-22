from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from functools import reduce
import torch


# given a MultiOmicDataset object, convert to Triplets (anchor,positive,negative)
class TripletMultiOmicDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mydataset, train = True):
        self.mydataset = mydataset
        self.train = train
        if self.train:
            self.train_data = self.mydataset
            self.labels_set, self.label_to_indices = self.get_label_indices(self.mydataset.y)
        else:
            self.test_data = self.mydataset
            self.test_triplets = self.generate_triplets(labels = self.mydataset.y, 
                                                        N = len(self.test_data))
    def __getitem__(self, index):
        if self.train:
            # get anchor sample and its label
            anchor, label = self.train_data[index][0], self.train_data[index][1].item()
            # choose another sample with same label
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label])
            # choose another sample with a different label 
            negative_label = np.random.choice(list(self.labels_set - set([label])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            pos = self.train_data[positive_index][0] # positive example
            neg = self.train_data[negative_index][0] # negative example
        else:
            anchor = self.test_data[self.test_triplets[index][0]][0]
            pos = self.test_data[self.test_triplets[index][1]][0]
            neg = self.test_data[self.test_triplets[index][2]][0]
            label = self.test_data[index][1].item()
        return anchor, pos, neg, label

    def __len__(self):
        return len(self.mydataset)
    
    def get_label_indices(self, labels):
        labels_set = set(labels.numpy())
        label_to_indices = {label: np.where(labels.numpy() == label)[0]
                             for label in labels_set}
        return labels_set, label_to_indices    

    def generate_triplets(self, labels, N, seed = 42):
        labels_set, label_to_indices = self.get_label_indices(labels)
        random_state = np.random.RandomState(seed)
        triplets = [[i,
                     random_state.choice(label_to_indices[labels[i].item()]),
                     random_state.choice(label_to_indices[
                         np.random.choice(
                             list(labels_set - set([labels[i].item()]))
                         )
                     ])
                    ] for i in range(N)]
        return triplets


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
        subset_dat = {x: self.dat[x][index] for x in self.dat.keys()}
        subset_y = self.y[index]
        return subset_dat, subset_y
    
    def __len__ (self):
        """Get the total number of samples in the dataset.

        Returns:
            An integer representing the number of samples in the dataset.
        """
        return len(self.y)
    
    def subset(self, indices):
        """Create a new dataset containing only the specified indices.

        Args:
            indices (list or np.array): A 1D array of indices to include in the subset.

        Returns:
            A new MultiomicDataset containing the specified subset of data.
        """
        indices = np.asarray(indices)
        subset_dat = {key: self.dat[key][indices] for key in self.dat.keys()}
        subset_y = np.asarray(self.y)[indices]
        subset_samples = np.asarray(self.samples)[indices]
        return MultiomicDataset(subset_dat, subset_y, self.features, subset_samples)
    
def get_labels(dat, drugs, drugName, batch_size, concatenate = False):
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
    if concatenate == True:
        dat = {'all': pd.concat(dat)}
    y = y[samples]
    return dat, y, samples

# dat: list of matrices, features on the rows, samples on the columns
# ann: pandas data frame with 'y' as variable for the corresponding samples on the matrix columns
# task_type: type of outcome (classification/regression)
# notice the tables are transposed to return samples on rows, features on columns
def get_torch_dataset(dat, labels, samples):
    # keep a copy of row/column names
    features = {x: dat[x].index for x in dat.keys()}
    dat = {x: torch.from_numpy(np.array(dat[x].T)).float() for x in dat.keys()}
    y =  torch.from_numpy(np.array(labels)).float()
    return MultiomicDataset(dat, y, features, samples)

def make_dataset(dat, *args, **kwargs):
    dat, y, samples = get_labels(dat, *args, **kwargs)
    dataset = get_torch_dataset(dat, y, samples)
    return dataset
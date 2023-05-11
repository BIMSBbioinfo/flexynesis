from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from functools import reduce
import torch
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from .feature_selection import filter_by_laplacian

from itertools import chain

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

# convert_to_labels: if true, given a numeric list, convert to binary labels by median value 
class DataImporter:
    def __init__(self, path, outcome_var, data_types, concatenate = False, min_features=None, top_percentile=None):
        self.path = path
        self.outcome_var = outcome_var
        self.data_types = data_types
        self.concatenate = concatenate
        self.min_features = min_features
        self.top_percentile = top_percentile
        # Initialize the label encoder
        self.label_encoder = None # only used of labels are categorical
        # initialize data scalers
        self.scalers = None
        # initialize data transformers
        self.transformers = None
        
    def read_data(self, folder_path, file_ext='.csv'):
        data = {}
        for file in os.listdir(folder_path):
            if file.endswith(file_ext):
                file_path = os.path.join(folder_path, file)
                file_name = os.path.splitext(file)[0]
                print("importing ",file_path)
                data[file_name] = pd.read_csv(file_path, index_col=0)
        return data
    
    def cleanup_data(self, df_dict, variance_threshold=1e-5):
        cleaned_dfs = {}
        sample_masks = []

        # First pass: remove near-zero-variation features and create masks for informative samples
        for key, df in df_dict.items():
            original_features_count = df.shape[0]

            # Step 1: Remove near-zero-variation features
            # Compute variances of features (along rows)
            feature_variances = df.var(axis=1)
            # Keep only features with variance above the threshold
            df = df.loc[feature_variances > variance_threshold, :]

            removed_features_count = original_features_count - df.shape[0]
            print(f"DataFrame {key} - Removed {removed_features_count} features.")

            # Step 2: Create masks for informative samples
            # Compute standard deviation of samples (along columns)
            sample_stdevs = df.std(axis=0)
            # Create mask for samples that do not have std dev of 0 or NaN
            mask = np.logical_and(sample_stdevs != 0, np.logical_not(np.isnan(sample_stdevs)))
            sample_masks.append(mask)

            cleaned_dfs[key] = df

        # Find samples that are informative in all dataframes
        common_mask = pd.DataFrame(sample_masks).all()

        # Second pass: apply common mask to all dataframes
        for key in cleaned_dfs.keys():
            original_samples_count = cleaned_dfs[key].shape[1]
            cleaned_dfs[key] = cleaned_dfs[key].loc[:, common_mask]
            removed_samples_count = original_samples_count - cleaned_dfs[key].shape[1]
            print(f"DataFrame {key} - Removed {removed_samples_count} samples ({removed_samples_count / original_samples_count * 100:.2f}%).")

        return cleaned_dfs

    def import_data(self):
        training_path = os.path.join(self.path, 'train')
        testing_path = os.path.join(self.path, 'test') 

        self.validate_data_folders(training_path, testing_path)
        
        training_data = self.read_data(training_path)
        testing_data = self.read_data(testing_path)

        # cleanup uninformative features/samples, assign labels, do feature selection on training data
        train_dat, train_y, train_samples = self.process_data(training_data, split = 'train')
        test_dat, test_y, test_samples = self.process_data(testing_data, split = 'test')
        
        # harmonize feature sets in train/test
        train_dat, test_dat = self.harmonize(train_dat, test_dat)
        
        # Normalize the training data (for testing data, use normalisation factors
        # learned from training data to apply on test data (see fit = False)
        train_dat = self.normalize_data(train_dat, scaler_type="standard", fit=True)
        test_dat = self.normalize_data(test_dat, scaler_type="standard", fit=False)
        
        # convert to pytorch datasets 
        training_dataset = self.get_torch_dataset(train_dat, train_y, train_samples)
        testing_dataset = self.get_torch_dataset(test_dat, test_y, test_samples)
       
        # for early fusion, concatenate all data matrices and feature lists 
        if self.concatenate:
            training_dataset.dat = {'all': torch.cat([training_dataset.dat[x] for x in training_dataset.dat.keys()], dim = 1)}
            training_dataset.features = {'all': list(chain(*training_dataset.features.values()))}
            
            testing_dataset.dat = {'all': torch.cat([testing_dataset.dat[x] for x in testing_dataset.dat.keys()], dim = 1)}
            testing_dataset.features = {'all': list(chain(*testing_dataset.features.values()))}
            
        return training_dataset, testing_dataset
    
    def validate_data_folders(self, training_path, testing_path):
        training_files = set(os.listdir(training_path))
        testing_files = set(os.listdir(testing_path))

        required_files = {'clin.csv'} | {f"{dt}.csv" for dt in self.data_types}

        if not required_files.issubset(training_files):
            missing_files = required_files - training_files
            raise ValueError(f"Missing files in training folder: {', '.join(missing_files)}")

        if not required_files.issubset(testing_files):
            missing_files = required_files - testing_files
            raise ValueError(f"Missing files in testing folder: {', '.join(missing_files)}")


    def process_data(self, data, split = 'train'):
        # remove uninformative features and samples with no information (from data matrices)
        dat = self.cleanup_data({x: data[x] for x in self.data_types})
        ann = data['clin']
        
        dat, y, samples = self.get_labels(dat, ann)
        
        # do feature selection: only applied to training data
        if split == 'train': 
            if self.min_features or self.top_percentile:
                dat = self.filter(dat, self.min_features, self.top_percentile)
        return dat, y, samples

    def get_labels(self, dat, ann):
        y = ann[self.outcome_var]
        y = y[~y.isna()]

        samples = list(reduce(set.intersection, [set(item) for item in [dat[x].columns for x in dat.keys()]]))
        samples = list(set(y.index).intersection(samples))
        dat = {x: dat[x][samples] for x in dat.keys()}
        y = y[samples]
        return dat, y, samples

    def encode_labels(self, labels):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels

    def get_torch_dataset(self, dat, labels, samples):
        features = {x: dat[x].index for x in dat.keys()}
        dat = {x: torch.from_numpy(np.array(dat[x].T)).float() for x in dat.keys()}

        # Check if the labels are categorical or numerical
        is_categorical = not np.issubdtype(np.array(labels).dtype, np.number)

        # Encode labels if they are categorical
        if is_categorical:
            labels = self.encode_labels(labels)

        y = torch.from_numpy(np.array(labels)).float()
        return MultiomicDataset(dat, y, features, samples)

    def normalize_data(self, data, scaler_type="standard", fit=True):
        print("normalizing data")
        # notice matrix transpositions during fit and finally after transformation
        # because data matrices have features on rows, 
        # while scaling methods assume features to be on the columns. 
        if fit:
            if scaler_type == "standard":
                self.scalers = {x: StandardScaler().fit(data[x].T) for x in data.keys()}
            elif scaler_type == "min_max":
                self.scalers = {x: MinMaxScaler().fit(data[x].T) for x in data.keys()}
            else:
                raise ValueError("Invalid scaler_type. Choose 'standard' or 'min_max'.")
        
        normalized_data = {x: pd.DataFrame(self.scalers[x].transform(data[x].T), 
                                           index=data[x].columns, 
                                           columns=data[x].index).T 
                           for x in data.keys()}
        return normalized_data
    
    def transform_data(self, data, transformation_type=None, fit=True):
        if fit:
            self.transformers = {}

        if transformation_type:
            if transformation_type == 'log':
                transformed_data = {x: np.log1p(data[x].T).T for x in data.keys()}
            elif transformation_type == 'sqrt':
                transformed_data = {x: np.sqrt(data[x].T).T for x in data.keys()}
            elif transformation_type == 'box-cox':
                transformed_data = {}
                for x in data.keys():
                    if fit:
                        pt = PowerTransformer(method='box-cox')
                        self.transformers[x] = pt.fit(data[x].T)
                    transformed_data[x] = pd.DataFrame(self.transformers[x].transform(data[x].T),
                                                        index=data[x].columns,
                                                        columns=data[x].index).T
            else:
                raise ValueError("Invalid transformation_type. Choose 'log', 'sqrt', or 'box-cox'.")
        else:
            transformed_data = data
        return transformed_data    

    def filter(self, dat, min_features, top_percentile):
        print("Filtering features based on Laplacian Score")
        counts = {x: max(int(dat[x].shape[0] * top_percentile), min_features) for x in dat.keys()}
        dat = {x: filter_by_laplacian(dat[x].T, topN=counts[x]).T for x in dat.keys()}
        return dat

    def harmonize(self, dat1, dat2):
        print("Harmonizing features between train and test")
        # Get common features
        common_features = {x: dat1[x].index.intersection(dat2[x].index) for x in self.data_types}
        # Subset both datasets to only include common features
        dat1 = {x: dat1[x].loc[common_features[x]] for x in dat1.keys()}
        dat2 = {x: dat2[x].loc[common_features[x]] for x in dat2.keys()}
        return dat1, dat2
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import download_url, extract_gz
from torch_geometric.data import Dataset as PYGDataset

import numpy as np
import pandas as pd
from functools import reduce
import torch
import os
import shutil

from tqdm import tqdm


from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from .feature_selection import filter_by_laplacian

from itertools import chain


# convert_to_labels: if true, given a numeric list, convert to binary labels by median value 
class DataImporter:
    """
    A class for importing, cleaning, and preprocessing multi-omic data for downstream analysis,
    including support for incorporating graph-based features from protein-protein interaction networks.

    Attributes:
        path (str): The base directory path where data is stored.
        data_types (list[str]): A list of data modalities to import (e.g., 'rna', 'methylation').
        log_transform (bool): If True, apply log transformation to the data.
        concatenate (bool): If True, concatenate features from different modalities.
        restrict_to_features (path): path to file that includes user-specific list of features (default: None)
        min_features (int): The minimum number of features to retain after filtering.
        top_percentile (float): The top percentile of features to retain based on variance.
        correlation_threshold(float): The correlation threshold for dropping highly redundant features
        variance_threshold (float): The variance threshold for removing low-variance features.
        na_threshold (float): The threshold for removing features with too many NA values.
        string_organism (int): STRING organism (species) id (default: 9606 (human)).
        string_node_name (str): The type of node names used in the graph. Available options: "gene_name", "gene_id" (default: "gene_name").
    Methods:
        import_data():
            The primary method to orchestrate the data import and preprocessing workflow. It follows these steps:
                1. Validates the presence of required data files in training and testing directories.
                2. Imports data using `read_data` for both training and testing sets.
                3. Cleans and preprocesses the data through `cleanup_data`.
                4. Processes data to align features and samples across modalities using `process_data`.
                5. Harmonizes training and testing datasets to have the same features using `harmonize`.
                6. Optionally applies log transformation.
                7. Normalizes the data.
                8. Encodes labels and prepares PyTorch datasets.
                9. Returns PyTorch datasets for training and testing.

        validate_data_folders(training_path, testing_path):
            Checks for the presence of required data files in specified directories.

        read_data(folder_path):
            Reads and imports data files for a given modality from a specified folder.

        cleanup_data(df_dict):
            Cleans dataframes by removing low-variance features, imputing missing values, 
            removing uninformative featuers (too many NA values).

        process_data(data, split='train'):
            Prepares the data for model input by cleaning, filtering, and selecting features and samples.

        select_features(dat):
            Implements an unsupervised feature selection by ranking features by the Laplacian score, keeping the features at 
            the top percentile range and removing highly redundant features (optional) based on a correlation threshold,
            while keeping a minimum number of top features as requested by the user. 

        harmonize(dat1, dat2):
            Aligns the feature sets of two datasets (e.g., training and testing) to have the same features.

        transform_data(data):
            Applies log transformation to the data matrices.

        normalize_data(data, scaler_type="standard", fit=True):
            Applies normalization to the data matrices.

        get_labels(dat, ann):
            Aligns and subsets annotations to match the samples present in the data matrices.

        get_torch_dataset(dat, ann, samples, feature_ann):
            Prepares and returns PyTorch datasets for the imported and processed data.

        encode_labels(df):
            Encodes categorical labels in the annotation dataframe.
    """

    def __init__(self, path, data_types, processed_dir="processed", log_transform = False, concatenate = False, restrict_to_features = None, min_features=None,
                 top_percentile=20, correlation_threshold = 0.9, variance_threshold=0.01, na_threshold=0.1, downsample=0):
        self.path = path
        self.data_types = data_types
        self.processed_dir = os.path.join(self.path, processed_dir)
        self.concatenate = concatenate
        self.min_features = min_features
        self.top_percentile = top_percentile
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.na_threshold = na_threshold
        self.log_transform = log_transform
        # Initialize a dictionary to store the label encoders
        self.encoders = {} # used if labels are categorical 
        # initialize data scalers
        self.scalers = None
        # initialize data transformers
        self.transformers = None
        self.downsample = downsample

        # read user-specified feature list to restrict the analysis to that
        self.restrict_to_features = restrict_to_features
        self.get_user_features()
        
        # for each feature in the input training data; keep a log of what happens to the feature 
        # record metrics such as laplacian score, variance
        # record if the feature is dropped due to these metrics or due to high correlation to a 
        # higher ranking feature
        self.feature_logs = {} 
    
    def get_user_features(self):
        """
        Load and process user-specified features from a file.
        """
        if self.restrict_to_features is not None:
            if not os.path.isfile(self.restrict_to_features):
                raise FileNotFoundError(f"File not found: {self.restrict_to_features}")
            try:
                with open(self.restrict_to_features, 'r') as fp:
                    # Read and process the file
                    feature_list = [x.strip() for x in fp.read().splitlines() if x.strip()]
                    # Ensure uniqueness and assign
                    self.restrict_to_features = np.unique(feature_list)
            except Exception as e:
                print(f"An error occurred while processing the file: {e}")
        else: 
            self.restrict_to_features = None

    def import_data(self):
        print("\n[INFO] ================= Importing Data =================")
        training_path = os.path.join(self.path, 'train')
        testing_path = os.path.join(self.path, 'test')

        self.validate_data_folders(training_path, testing_path)
        
        # raw data matrices as exists in the data path
        train_dat = self.read_data(training_path)
        test_dat = self.read_data(testing_path)
        
        if self.downsample > 0:
            print("[INFO] Randomly drawing",self.downsample,"samples for training")
            train_dat = self.subsample(train_dat, self.downsample)

        if self.restrict_to_features is not None:
            train_dat = self.filter_by_features(train_dat, self.restrict_to_features)
            test_dat = self.filter_by_features(test_dat, self.restrict_to_features)
            
        # check for any problems with the the input files 
        self.validate_input_data(train_dat, test_dat)

        # cleanup uninformative features/samples, subset annotation data, do feature selection on training data
        train_dat, train_ann, train_samples, train_features = self.process_data(train_dat, split = 'train')
        test_dat, test_ann, test_samples, test_features = self.process_data(test_dat, split = 'test')
        
        # harmonize feature sets in train/test
        train_dat, test_dat = self.harmonize(train_dat, test_dat)

        train_feature_ann = {}
        test_feature_ann = {}

        # log_transform 
        if self.log_transform:
            print("[INFO] transforming data to log scale")
            train_dat = self.transform_data(train_dat)
            test_dat = self.transform_data(test_dat)
        
        # Normalize the training data (for testing data, use normalisation factors
        # learned from training data to apply on test data (see fit = False)
        train_dat = self.normalize_data(train_dat, scaler_type="standard", fit=True)
        test_dat = self.normalize_data(test_dat, scaler_type="standard", fit=False)

        # encode the variable annotations, convert data matrices and annotations pytorch datasets 
        training_dataset = self.get_torch_dataset(train_dat, train_ann, train_samples, train_feature_ann)
        testing_dataset = self.get_torch_dataset(test_dat, test_ann, test_samples, test_feature_ann)

        # NOTE: Exporting to the disk happens in get_torch_dataset, so the concatenate doesn't work.
        # TODO: Find better way for early integration, or move it to get_torch_dataset. Otherwise it will be ignored.
        # for early fusion, concatenate all data matrices and feature lists
        if self.concatenate:
            training_dataset.dat = {'all': torch.cat([training_dataset.dat[x] for x in training_dataset.dat.keys()], dim = 1)}
            training_dataset.features = {'all': list(chain(*training_dataset.features.values()))}

            testing_dataset.dat = {'all': torch.cat([testing_dataset.dat[x] for x in testing_dataset.dat.keys()], dim = 1)}
            testing_dataset.features = {'all': list(chain(*testing_dataset.features.values()))}
        
        print("[INFO] Training Data Stats: ", training_dataset.get_dataset_stats())
        print("[INFO] Test Data Stats: ", testing_dataset.get_dataset_stats())
        print("[INFO] Merging Feature Logs...")
        logs = self.feature_logs
        self.feature_logs = {x: pd.merge(logs['cleanup'][x], 
                                         logs['select_features'][x], 
                                         on = 'feature', how = 'outer', 
                                         suffixes=['_cleanup', '_laplacian']) for x in self.data_types}
        print("[INFO] Data import successful.")
        
        return training_dataset, testing_dataset
            
    def validate_data_folders(self, training_path, testing_path):
        print("[INFO] Validating data folders...")
        training_files = set(os.listdir(training_path))
        testing_files = set(os.listdir(testing_path))

        required_files = {'clin.csv'} | {f"{dt}.csv" for dt in self.data_types}

        if not required_files.issubset(training_files):
            missing_files = required_files - training_files
            raise ValueError(f"Missing files in training folder: {', '.join(missing_files)}")

        if not required_files.issubset(testing_files):
            missing_files = required_files - testing_files
            raise ValueError(f"Missing files in testing folder: {', '.join(missing_files)}")
    
    def read_data(self, folder_path):
        data = {}
        required_files = {'clin.csv'} | {f"{dt}.csv" for dt in self.data_types}
        print("\n[INFO] ----------------- Reading Data ----------------- ")
        for file in required_files:
            file_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]
            print(f"[INFO] Importing {file_path}...")
            data[file_name] = pd.read_csv(file_path, index_col=0)
        return data
    
    # randomly draw N samples; return subset of dat (output of read_data)
    def subsample(self, dat, N):
        clin = dat['clin'].sample(N)
        dat_sub = {x: dat[x][clin.index] for x in self.data_types}
        dat_sub['clin'] = clin
        return dat_sub


    def filter_by_features(self, dat, features):
        """
        If the user has provided list of features to restrict the analysis to, 
        subset train/test data to only include those features
        """
        dat_filtered = {
            key: df if key == "clin" else df.loc[df.index.intersection(features)]
            for key, df in dat.items()
        }

        print("[INFO] The initial features are filtered to include user-provided features only")
        for key, df in dat_filtered.items():
            remaining_features = len(df.index)
            print(f"In layer '{key}', {remaining_features} features are remaining after filtering.")
        return dat_filtered

    def process_data(self, data, split = 'train'):
        print(f"\n[INFO] ----------------- Processing Data ({split}) ----------------- ")
        # remove uninformative features and samples with no information (from data matrices)
        dat = self.cleanup_data({x: data[x] for x in self.data_types})
        ann = data['clin']
        dat, ann, samples = self.get_labels(dat, ann)
        # do feature selection: only applied to training data
        if split == 'train': 
            if self.top_percentile:
                dat = self.select_features(dat)
        features = {x: dat[x].index for x in dat.keys()}
        return dat, ann, samples, features

    def cleanup_data(self, df_dict):
        print("\n[INFO] ----------------- Cleaning Up Data ----------------- ")
        cleaned_dfs = {}
        sample_masks = []

        feature_logs = {} # keep track of feature variation/NA value scores 
        # First pass: remove near-zero-variation features and create masks for informative samples
        for key, df in df_dict.items():
            print("\n[INFO] working on layer: ",key)
            original_features_count = df.shape[0]

            # Compute variances and NA percentages for each feature in the DataFrame
            feature_variances = df.var(axis=1)
            na_percentages = df.isna().mean(axis=1)

            # Combine variances and NA percentages into a single DataFrame for logging
            log_df = pd.DataFrame({ 'feature': df.index, 'na_percent': na_percentages, 'variance': feature_variances, 'selected': False})
            
            # Filter based on both variance and NA percentage thresholds
            # Identify features that meet both criteria
            df = df.loc[(feature_variances > feature_variances.quantile(self.variance_threshold)) & (na_percentages < self.na_threshold)]
            # set selected features to True
            log_df['selected'] = (log_df['variance'] > feature_variances.quantile(self.variance_threshold)) & (log_df['na_percent'] < self.na_threshold)
            feature_logs[key] = log_df
            
            # Step 3: Fill NA values with the median of the feature
            # Check if there are any NA values in the DataFrame
            
            if np.sum(df.isna().sum()) > 0:
                missing_rows = df.isna().any(axis=1)
                print("[INFO] Imputing NA values to median of features, affected # of cells in the matrix", np.sum(df.isna().sum()), " # of rows:",sum(missing_rows))
                
                # Calculate medians for each 'column' (originally rows) and fill NAs
                # Note: After transposition, operations are more efficient
                df_T = df.T
                medians_T = df_T.median(axis=0)
                df_T.fillna(medians_T, inplace=True)
                df = df_T.T
            
            print("[INFO] Number of NA values: ",np.sum(df.isna().sum()))
                                   
            removed_features_count = original_features_count - df.shape[0]
            print(f"[INFO] DataFrame {key} - Removed {removed_features_count} features.")
        
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
            print(f"[INFO] DataFrame {key} - Removed {removed_samples_count} samples ({removed_samples_count / original_samples_count * 100:.2f}%).")

        # update feature logs from this process
        self.feature_logs['cleanup'] = feature_logs
        return cleaned_dfs

    def get_labels(self, dat, ann):
        # subset samples and reorder annotations for the samples 
        samples = list(reduce(set.intersection, [set(item) for item in [dat[x].columns for x in dat.keys()]]))
        samples = list(set(ann.index).intersection(samples))
        dat = {x: dat[x][samples] for x in dat.keys()}
        ann = ann.loc[samples]
        return dat, ann, samples

    # unsupervised feature selection using laplacian score and correlation filters (optional)
    def select_features(self, dat):
        counts = {x: max(int(dat[x].shape[0] * self.top_percentile / 100), self.min_features) for x in dat.keys()}
        dat_filtered = {}
        feature_logs = {} # feature log for each layer
        for layer in dat.keys():
            # filter features in the layer and keep a log of filtering process; notice we provide a transposed matrix
            X_filt, log_df = filter_by_laplacian(X = dat[layer].T, layer = layer, 
                                                      topN=counts[layer], correlation_threshold = self.correlation_threshold)
            dat_filtered[layer] = X_filt.T # transpose after laplacian filtering again
            feature_logs[layer] = log_df
        # update main feature logs with events from this function
        self.feature_logs['select_features'] = feature_logs
        return dat_filtered 

    def harmonize(self, dat1, dat2):
        print("\n[INFO] ----------------- Harmonizing Data Sets ----------------- ")
        # Get common features
        common_features = {x: dat1[x].index.intersection(dat2[x].index) for x in self.data_types}
        # Subset both datasets to only include common features
        dat1 = {x: dat1[x].loc[common_features[x]] for x in dat1.keys()}
        dat2 = {x: dat2[x].loc[common_features[x]] for x in dat2.keys()}
        print("\n[INFO] ----------------- Finished Harmonizing ----------------- ")

        return dat1, dat2
    
    def transform_data(self, data):
        transformed_data = {x: np.log1p(data[x].T).T for x in data.keys()}
        return transformed_data    

    def normalize_data(self, data, scaler_type="standard", fit=True):
        print("\n[INFO] ----------------- Normalizing Data ----------------- ")
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
    
    def get_torch_dataset(self, dat, ann, samples, feature_ann):
        
        features = {x: dat[x].index for x in dat.keys()}
        dat = {x: torch.from_numpy(np.array(dat[x].T)).float() for x in dat.keys()}

        ann, variable_types, label_mappings = self.encode_labels(ann)

        # Convert DataFrame to tensor
        ann = {col: torch.from_numpy(ann[col].values) for col in ann.columns}
        return MultiOmicDataset(dat, ann, variable_types, features, samples, label_mappings)

    def encode_labels(self, df):
        label_mappings = {}
        def encode_column(series):
            nonlocal label_mappings  # Declare as nonlocal so that we can modify it
            # Fill NA values with 'missing' 
            # series = series.fillna('missing')
            if series.name not in self.encoders:
                self.encoders[series.name] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoded_series = self.encoders[series.name].fit_transform(series.to_frame())
            else:
                encoded_series = self.encoders[series.name].transform(series.to_frame())
            
            # also save label mappings 
            label_mappings[series.name] = {
                    int(code): label for code, label in enumerate(self.encoders[series.name].categories_[0])
                }
            return encoded_series.ravel()

        # Select only the categorical columns
        df_categorical = df.select_dtypes(include=['object', 'category']).apply(encode_column)

        # Combine the encoded categorical data with the numerical data
        df_encoded = pd.concat([df.select_dtypes(exclude=['object', 'category']), df_categorical], axis=1)

        # Store the variable types
        variable_types = {col: 'categorical' for col in df_categorical.columns}
        variable_types.update({col: 'numerical' for col in df.select_dtypes(exclude=['object', 'category']).columns})

        return df_encoded, variable_types, label_mappings

    def validate_input_data(self, train_dat, test_dat):   
        print("\n[INFO] ----------------- Checking for problems with the input data ----------------- ")
        errors = []
        warnings = []
        def check_rownames(dat, split):
            # Check 1: Validate first columns are unique
            for file_name, df in dat.items():
                if not df.index.is_unique:
                    identifier_type = "Sample labels" if file_name == 'clin' else "Feature names"
                    errors.append(f"Error in {split}/{file_name}.csv: {identifier_type} in the first column must be unique.")

        def check_sample_labels(dat, split):
            clin_samples = set(dat['clin'].index)
            for file_name, df in dat.items():
                if file_name != 'clin':
                    omics_samples = set(df.columns)
                    matching_samples = clin_samples.intersection(omics_samples)
                    if not matching_samples:
                        errors.append(f"Error: No matching sample labels found between {split}/clin.csv and {split}/{file_name}.csv.")
                    elif len(matching_samples) < len(clin_samples):
                        missing_samples = clin_samples - matching_samples
                        warnings.append(f"Warning: Some sample labels in {split}/clin.csv are missing in {split}/{file_name}.csv: {missing_samples}")

        def check_common_features(train_dat, test_dat):
            for file_name in train_dat:
                if file_name != 'clin' and file_name in test_dat:
                    train_features = set(train_dat[file_name].index)
                    test_features = set(test_dat[file_name].index)
                    common_features = train_features.intersection(test_features)
                    if not common_features:
                        errors.append(f"Error: No common features found between train/{file_name}.csv and test/{file_name}.csv.")

        check_rownames(train_dat, 'train')
        check_rownames(test_dat, 'test')

        check_sample_labels(train_dat, 'train')
        check_sample_labels(test_dat, 'test')

        check_common_features(train_dat, test_dat)

        # Handle errors and warnings
        if warnings:
            print("\n[WARNING] Warnings:\n")
            for i, warning in enumerate(warnings, 1):
                print(f"[WARNING] {i}. {warning}")

        if errors:
            print("[INFO] Found problems with the input data:\n")
            for i, error in enumerate(errors, 1):
                print(f"[ERROR] {i}. {error}")
            raise Exception("[ERROR] Please correct the above errors and try again.")


        if not warnings and not errors:
            print("[INFO] Data structure is valid with no errors or warnings.")       
            

class MultiOmicDataset(Dataset):
    """A PyTorch dataset for multiomic data.

    Args:
        dat (dict): A dictionary with keys corresponding to different types of data and values corresponding to matrices of the same shape. All matrices must have the same number of samples (rows).
        ann (data.frame): Data frame with samples on the rows, sample annotations on the columns 
        features (list or np.array): A 1D array of feature names with length equal to the number of columns in each matrix.
        samples (list or np.array): A 1D array of sample names with length equal to the number of rows in each matrix.

    Returns:
        A PyTorch dataset that can be used for training or evaluation.
    """

    def __init__(self, dat, ann, variable_types, features, samples, label_mappings, feature_ann=None):
        """Initialize the dataset."""
        self.dat = dat
        self.ann = ann
        self.variable_types = variable_types
        self.features = features
        self.samples = samples
        self.label_mappings = label_mappings
        self.feature_ann = feature_ann or {}

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
        subset_ann = {x: self.ann[x][index] for x in self.ann.keys()}
        return subset_dat, subset_ann
    
    def __len__ (self):
        """Get the total number of samples in the dataset.

        Returns:
            An integer representing the number of samples in the dataset.
        """
        return len(self.samples)
    
    def subset(self, indices):
            """Create a new dataset object containing only the specified indices.

            Args:
                indices (list of int): The indices of the samples to include in the subset.

            Returns:
                MultiOmicDataset: A new dataset object with the same structure but only containing the selected samples.
            """
            subset_dat = {x: self.dat[x][indices] for x in self.dat.keys()}
            subset_ann = {x: self.ann[x][indices] for x in self.ann.keys()}
            subset_samples = [self.samples[idx] for idx in indices]

            # Create a new dataset object
            return MultiOmicDataset(subset_dat, subset_ann, self.variable_types, self.features,
                                    subset_samples, self.label_mappings, self.feature_ann)
    
    def get_feature_subset(self, feature_df):
        """Get a subset of data matrices corresponding to specified features and concatenate them into a pandas DataFrame.

        Args:
            feature_df (pandas.DataFrame): A DataFrame which contains at least two columns: 'layer' and 'name'. 

        Returns:
            A pandas DataFrame that concatenates the data matrices for the specified features from all layers. 
        """
        # Convert the DataFrame to a dictionary
        feature_dict = feature_df.groupby('layer')['name'].apply(list).to_dict()

        dfs = []
        for layer, features in feature_dict.items():
            if layer in self.dat:
                # Create a dictionary to look up indices by feature name for each layer
                feature_index_dict = {feature: i for i, feature in enumerate(self.features[layer])}
                # Get the indices for the requested features
                indices = [feature_index_dict[feature] for feature in features if feature in feature_index_dict]
                # Subset the data matrix for the current layer using the indices
                subset = self.dat[layer][:, indices]
                # Convert the subset to a pandas DataFrame, add the layer name as a prefix to each column name
                df = pd.DataFrame(subset, columns=[f'{layer}_{feature}' for feature in features if feature in feature_index_dict])
                dfs.append(df)
            else:
                print(f"Layer {layer} not found in the dataset.")

        # Concatenate the dataframes along the columns axis
        result = pd.concat(dfs, axis=1)

        # Set the sample names as the row index
        result.index = self.samples

        return result
    
    def get_dataset_stats(self):
        stats = {': '.join(['feature_count in', x]): self.dat[x].shape[1] for x in self.dat.keys()}
        stats['sample_count'] = len(self.samples)
        return(stats)
    

# given a MultiOmicDataset object, convert to Triplets (anchor,positive,negative)
class TripletMultiOmicDataset(Dataset):
    """
    For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, mydataset, main_var):
        self.dataset = mydataset
        self.main_var = main_var
        self.labels_set, self.label_to_indices = self.get_label_indices(self.dataset.ann[self.main_var])
    def __getitem__(self, index):
        # get anchor sample and its label
        anchor, y_dict = self.dataset[index][0], self.dataset[index][1] 
        # choose another sample with same label
        label = y_dict[self.main_var].item()
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label])
        # choose another sample with a different label 
        negative_label = np.random.choice(list(self.labels_set - set([label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        pos = self.dataset[positive_index][0] # positive example
        neg = self.dataset[negative_index][0] # negative example
        return anchor, pos, neg, y_dict

    def __len__(self):
        return len(self.dataset)
    
    def get_label_indices(self, labels):
        labels_set = set(labels.numpy())
        label_to_indices = {label: np.where(labels.numpy() == label)[0]
                             for label in labels_set}
        return labels_set, label_to_indices

class MultiOmicDatasetNW(Dataset):
    def __init__(self, multiomic_dataset, interaction_df):
        self.multiomic_dataset = multiomic_dataset
        self.interaction_df = interaction_df
        
        # Precompute common features and edge index
        self.common_features = self.find_common_features()
        self.gene_to_index = {gene: idx for idx, gene in enumerate(self.common_features)}
        self.edge_index = self.create_edge_index()
        self.samples = self.multiomic_dataset.samples
        self.variable_types = self.multiomic_dataset.variable_types
        self.label_mappings = self.multiomic_dataset.label_mappings
        self.ann = self.multiomic_dataset.ann
        
        # Precompute all node features for all samples
        self.node_features_tensor = self.precompute_node_features()
        
        # Store labels for all samples
        self.labels = {target_name: labels for target_name, labels in self.multiomic_dataset.ann.items()}
        
    def find_common_features(self):
        common_features = set.intersection(*(set(features) for features in self.multiomic_dataset.features.values()))
        interaction_genes = set(self.interaction_df['protein1']).union(set(self.interaction_df['protein2']))
        return list(common_features.intersection(interaction_genes))

    def create_edge_index(self):
        filtered_df = self.interaction_df[
            (self.interaction_df['protein1'].isin(self.common_features)) & 
            (self.interaction_df['protein2'].isin(self.common_features))
        ]
        edge_list = [(self.gene_to_index[row['protein1']], self.gene_to_index[row['protein2']]) for index, row in filtered_df.iterrows()]
        return torch.tensor(edge_list, dtype=torch.long).t()

    def precompute_node_features(self):
        # Find indices of common features in each data matrix
        feature_indices = {data_type: [self.multiomic_dataset.features[data_type].get_loc(gene) 
                                        for gene in self.common_features]
                           for data_type in self.multiomic_dataset.dat}
        # Create a tensor to store all features [num_samples, num_nodes, num_data_types]
        num_samples = len(self.samples)
        num_nodes = len(self.common_features)
        num_data_types = len(self.multiomic_dataset.dat)
        all_features = torch.empty((num_samples, num_nodes, num_data_types), dtype=torch.float)

        # Extract features for each data type and place them in the tensor
        for i, data_type in enumerate(self.multiomic_dataset.dat):
            # Get the data matrix
            data_matrix = self.multiomic_dataset.dat[data_type]
            # Use advanced indexing to extract features for all samples at once
            indices = feature_indices[data_type]
            if indices:  # Ensure there are common features in this data type
                all_features[:, :, i] = data_matrix[:, indices]
        return all_features
        
    def __getitem__(self, idx):
        node_features_tensor = self.node_features_tensor[idx]
        y_dict = {target_name: self.labels[target_name][idx] for target_name in self.labels}
        return node_features_tensor, y_dict, self.samples[idx]

    def __len__(self):
        return len(self.samples)
    
    def print_stats(self):
        """
        Prints various statistics about the graph.
        """
        num_nodes = len(self.common_features)
        num_edges = self.edge_index.size(1)
        num_node_features = self.node_features_tensor.size(2)

        # Calculate degree for each node
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.index_add_(0, self.edge_index[0], torch.ones_like(self.edge_index[0]))
        degrees.index_add_(0, self.edge_index[1], torch.ones_like(self.edge_index[1]))  # For undirected graphs

        num_singletons = torch.sum(degrees == 0).item()
        non_singletons = degrees[degrees > 0]

        mean_edges_per_node = non_singletons.float().mean().item() if len(non_singletons) > 0 else 0
        median_edges_per_node = non_singletons.median().item() if len(non_singletons) > 0 else 0
        max_edges = degrees.max().item()

        print("Dataset Statistics:")
        print(f"Number of nodes: {num_nodes}")
        print(f"Total number of edges: {num_edges}")
        print(f"Number of node features per node: {num_node_features}")
        print(f"Number of singletons (nodes with no edges): {num_singletons}")
        print(f"Mean number of edges per node (excluding singletons): {mean_edges_per_node:.2f}")
        print(f"Median number of edges per node (excluding singletons): {median_edges_per_node}")
        print(f"Max number of edges per node: {max_edges}")
        
class STRING(PYGDataset):
    base_folder = "STRING"
    version = "12.0"
    files = ("links", "aliases")
    url = ("https://stringdb-downloads.org/download/"
           "protein.{data}.v{version}/"
           "{organism}.protein.{data}.v{version}.txt.gz")

    def __init__(self, root: str, organism: int = 9606, node_name: str = "gene_name") -> None:
        self.organism = organism
        self.node_name = node_name
        super().__init__(os.path.join(root, self.base_folder))
        
        if not os.path.exists(self.processed_paths[0]):
            self.download()
            self.process()
        self.graph_df = pd.read_csv(self.processed_paths[0], sep=",", header=0, index_col=0)

    def len(self) -> int:
        return 0

    def get(self, idx: int):
        return None

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.organism}.protein.{f}.v{self.version}.txt" for f in self.files]

    @property
    def processed_file_names(self) -> str:
        return "graph.csv"

    def download(self) -> None:
        folder = os.path.join(self.root, str(self.organism))
        for d in self.files:
            url = self.url.format(organism=self.organism, data=d, version=self.version)
            path = download_url(url, folder)
            extract_gz(path, folder)
            os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        graph_df = read_stringdb_graph(self.node_name, self.raw_paths[0], self.raw_paths[1])
        graph_df.dropna().to_csv(self.processed_paths[0])
        self.graph_df = graph_df  # Storing the DataFrame in the instance


def read_user_graph(fpath, sep=" ", header=None, **pd_read_csv_kw):
    """Read edge list from a file prepared by user.

    Returns
        two cols pandas df.
    """
    return pd.read_csv(fpath, sep=sep, header=header, **pd_read_csv_kw)

def read_stringdb_links(fname, top_neighbors = 5):
    """
    Reads and processes a STRING database file to extract and rank protein-protein interactions.

    The function loads a STRING database interactions file, filters out interactions with a combined
    score lower than 400, and then finds the top `top_neighbors` interactions for each protein based
    on the combined score. Each interaction is considered bidirectionally, ensuring that all possible
    interactions are accounted for without duplication.

    Args:
        fname (str): The file name or path to the STRING database file. The file should be a space-separated
            value format with a header row, and at least the columns 'protein1', 'protein2', and 'combined_score'.
        top_neighbors (int, optional): The number of top interactions to return for each protein. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame with columns 'protein1' and 'protein2' representing the top interactions
            for each protein. Each protein and partner identifier is simplified by stripping to the last
            element after splitting by ".".

    Example:
        >>> read_stringdb_links('string_interactions.txt', top_neighbors=3)
        # This will return a DataFrame with the top 3 interactions for each protein,
        # based on the 'combined_score', from the 'string_interactions.txt' file.
    """
    df = pd.read_csv(fname, header=0, sep=" ")
    df = df[df.combined_score > 400]
    df_expanded = pd.concat([
        df.rename(columns={'protein1': 'protein', 'protein2': 'partner'}),
        df.rename(columns={'protein2': 'protein', 'protein1': 'partner'})
    ])
    # Sort the expanded DataFrame by 'combined_score' in descending order
    df_expanded_sorted = df_expanded.sort_values(by='combined_score', ascending=False)
        # Reduce to unique interactions to avoid counting duplicates
    df_expanded_unique = df_expanded_sorted.drop_duplicates(subset=['protein', 'partner'])
    top_interactions = df_expanded_unique.groupby('protein').head(top_neighbors)
    df = top_interactions.rename(columns={'protein': 'protein1', 'partner': 'protein2'})
    df[["protein1", "protein2"]] = df[["protein1", "protein2"]].map(lambda a: a.split(".")[-1])
    return df

def read_stringdb_aliases(fname: str, node_name: str) -> dict[str, str]:
    if node_name == "gene_id":
        source = ("Ensembl_HGNC_ensembl_gene_id", "Ensembl_gene")
    elif node_name == "gene_name":
        source = ("Ensembl_EntrezGene", "Ensembl_HGNC_symbol")
    else:
        raise NotImplementedError
    protein_id_to_gene_id = {}
    with open(fname, "r") as f:
        next(f)
        for line in f:
            data = line.split()
            if data[-1].endswith(source[0]):
                protein_id_to_gene_id[data[0].split(".")[1]] = data[1]
            elif data[-1].endswith(source[1]):
                # TODO: Check here if the values are the same
                if protein_id_to_gene_id.get(data[0].split(".")[1], None) is None:
                    protein_id_to_gene_id[data[0].split(".")[1]] = data[1]
                else:
                    continue
            else:
                continue
    return protein_id_to_gene_id


def read_stringdb_graph(node_name, edges_data_path, nodes_data_path):
    # Read graph from the file
    graph_df = read_stringdb_links(edges_data_path)
    # Convert graph nodes names accordingly
    if node_name in ("gene_name", "gene_id"):
        node_name_mapping = read_stringdb_aliases(nodes_data_path, node_name)
    else:
        raise NotImplementedError("Node name must be either 'gene_name' or 'gene_id'.")

    def fn(a):
        try:
            out = node_name_mapping[a]
        except KeyError:
            # If smth wrong use NA.
            out = pd.NA
        return out

    graph_df[["protein1", "protein2"]] = graph_df[["protein1", "protein2"]].map(fn)
    return graph_df

def stringdb_links_to_list(df):
    lst = df[["protein1", "protein2"]].to_numpy().tolist()
    return lst


def split_by_median(tensor_dict):
    new_dict = {}
    for key, tensor in tensor_dict.items():
        # Check if the tensor is of a floating point type (i.e., it's numerical)
        if tensor.dtype in {torch.float16, torch.float32, torch.float64}:
            # Remove NaNs and compute median
            tensor_no_nan = tensor[torch.isfinite(tensor)]
            median_val = tensor_no_nan.sort().values[tensor_no_nan.numel() // 2]
            
            # Convert to categorical, but preserve NaNs
            tensor_cat = (tensor > median_val).float()
            tensor_cat[torch.isnan(tensor)] = float('nan')
            new_dict[key] = tensor_cat
        else:
            # If tensor is not numerical, leave it as it is
            new_dict[key] = tensor
    return new_dict

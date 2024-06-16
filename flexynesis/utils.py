from lightning import seed_everything
import pandas as pd
import numpy as np
import torch
import math
import requests
import tarfile
import os
from glob import glob
import re
import logging 
from tqdm import tqdm

from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.stats import pearsonr, linregress

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import community as community_louvain

def plot_dim_reduced(matrix, labels, method='pca', color_type='categorical', scatter_kwargs=None, legend_kwargs=None, figsize=(10, 8)):
    """
    Plots the first two dimensions of the transformed input matrix in a 2D scatter plot,
    with points colored based on the provided labels. The transformation method can be either PCA or UMAP.
    
    This function allows users to control several aspects of the plot such as the figure size, scatter plot properties, and legend properties.

    Args:
        matrix (np.array): Input data matrix (n_samples, n_features).
        labels (list): List of labels (strings or integers).
        method (str): Transformation method ('pca' or 'umap'). Default is 'pca'.
        color_type (str): Type of the color scale ('categorical' or 'numerical'). Default is 'categorical'.
        scatter_kwargs (dict, optional): Additional keyword arguments for plt.scatter. Default is None.
        legend_kwargs (dict, optional): Additional keyword arguments for plt.legend. Default is None.
        figsize (tuple): Size of the figure (width, height). Default is (10, 8).
    """
    
    plt.figure(figsize=figsize)
    
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}
    legend_kwargs = legend_kwargs if legend_kwargs else {}

    # Compute transformation
    if method.lower() == 'pca':
        transformer = PCA(n_components=2)
    elif method.lower() == 'umap':
        transformer = UMAP(n_components=2)
    else:
        raise ValueError("Invalid method. Expected 'pca' or 'umap'")
        
    transformed_matrix = transformer.fit_transform(matrix)

    # Create a pandas DataFrame for easier plotting
    transformed_df = pd.DataFrame(transformed_matrix, columns=[f"{method.upper()}1", f"{method.upper()}2"])

    if color_type == 'numerical':
        # Convert to numerical values, handling non-numeric and missing values
        labels = np.array([float(x) if not pd.isnull(x) and x not in {'nan', 'None'} else np.nan for x in labels])
    elif color_type == 'categorical':
        # Convert all to strings, handling missing values distinctly if necessary
        labels = np.array([str(x) if not pd.isnull(x) and x not in {'nan', 'None'} else 'Missing' for x in labels])
    else:
        raise ValueError("Invalid color_type specified. Must be 'numerical' or 'categorical'.")
    #labels = [-1 if pd.isnull(x) or x in {'nan', 'None'} else x for x in labels]

    # Add the labels to the DataFrame
    transformed_df["Label"] = labels

    if color_type == 'categorical':
        unique_labels = sorted(set(labels))
        colormap = plt.get_cmap("tab20", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            plt.scatter(
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}1"],
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}2"],
                color=colormap(i),
                label=label,
                **scatter_kwargs
            )
        if method.lower() == 'pca':
            plt.xlabel(f"PC1 (explained variance: {transformer.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14)
            plt.ylabel(f"PC2 (explained variance: {transformer.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
        else:
            plt.xlabel(f"{method.upper()} Dimension 1", fontsize=14)
            plt.ylabel(f"{method.upper()} Dimension 2", fontsize=14)

        plt.title(f"{method.upper()} Scatter Plot with Colored Labels", fontsize=18)
        plt.legend(title="Labels", **legend_kwargs)
    elif color_type == 'numerical':
        sc = plt.scatter(transformed_df[f"{method.upper()}1"], transformed_df[f"{method.upper()}2"], 
                         c=labels, **scatter_kwargs)
        plt.colorbar(sc, label='Label')
    plt.show()


def plot_scatter(true_values, predicted_values):
    """
    Plots a scatterplot of true vs predicted values, with a regression line and annotated with the Pearson correlation coefficient.

    Args:
        true_values (list or np.array): True values
        predicted_values (list or np.array): Predicted values
    """
    # Convert to numpy arrays (if not already)
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # Filter out NaN values
    not_nan_indices = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    true_values = true_values[not_nan_indices]
    predicted_values = predicted_values[not_nan_indices]

    # Calculate correlation coefficient
    corr, _ = pearsonr(true_values, predicted_values)
    corr_text = f"Pearson r: {corr:.2f}"
    
    # Generate scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.5)
    
    # Add regression line
    m, b = np.polyfit(true_values, predicted_values, 1)
    plt.plot(true_values, m*true_values + b, color='red')
    
    # Add correlation text
    plt.text(min(true_values), max(predicted_values), corr_text, fontsize=12, ha='left', va='top')
    
    # Add labels and title
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    plt.show()
    
    
def plot_boxplot(categorical_x, numerical_y, title_x = 'Categories', title_y = 'Values'):
    df = pd.DataFrame({title_x: categorical_x, title_y: numerical_y})
    # Create a boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(x=title_x, y=title_y, data=df, palette='Set2')
    plt.show()
    
    
# given a vector of numerical values which may contain 
# NAN values, return a binary grouping based on median values 
def split_by_median(v):
    return ((v - torch.nanmedian(v)) > 0).float()
    
def evaluate_survival(outputs, durations, events):
    """
    Computes the concordance index (c-index) for survival predictions.

    Parameters:
    - durations: A numpy array or a torch tensor of true survival times or durations.
    - events: A numpy array or a torch tensor indicating whether an event (e.g., death) occurred.
    - risk_scores: Predicted risk scores from the model. Higher scores should indicate higher risk of event.

    Returns:
    - A dictionary containing the c-index.
    """
    valid_indices = ~torch.isnan(durations) & ~torch.isnan(events)
    if valid_indices.sum() > 0:
        outputs = outputs[valid_indices]
        events = events[valid_indices]
        durations = durations[valid_indices]
    # Ensure inputs are in the correct format (numpy arrays)
    if isinstance(durations, torch.Tensor):
        durations = durations.numpy()
    if isinstance(events, torch.Tensor):
        events = events.numpy()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
    
    # Compute the c-index
    # reverse the directionality of risk_scores to make it compatible with lifelines' assumption
    c_index = concordance_index(durations, -outputs, events)
    return {'cindex': c_index}

def evaluate_classifier(y_true, y_pred, print_report = False):
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    # F1 score (macro)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    # Full classification report
    if print_report:
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
    return {"balanced_acc": balanced_acc, "f1_score": f1, "kappa": kappa}

def evaluate_regressor(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    slope, intercept, r_value, p_value, std_err = linregress(y_true,y_pred)
    r2 = r_value**2 
    return {"mse": mse, "r2": r2, "pearson_corr": r_value}

def evaluate_wrapper(y_pred_dict, dataset, surv_event_var = None, surv_time_var = None):
    metrics_list = []
    for var in y_pred_dict.keys():
        if dataset.variable_types[var] == 'numerical':
            if var == surv_event_var:
                events = dataset.ann[surv_event_var]
                durations = dataset.ann[surv_time_var]
                metrics = evaluate_survival(y_pred_dict[var], durations, events)
            else:
                ind = ~torch.isnan(dataset.ann[var])
                metrics = evaluate_regressor(dataset.ann[var][ind], y_pred_dict[var][ind].flatten())
        else:
            ind = ~torch.isnan(dataset.ann[var])
            metrics = evaluate_classifier(dataset.ann[var][ind], y_pred_dict[var][ind])

        for metric, value in metrics.items():
            metrics_list.append({
                'var': var,
                'variable_type': dataset.variable_types[var],
                'metric': metric,
                'value': value
            })
    # Convert the list of metrics to a DataFrame
    return pd.DataFrame(metrics_list)

def get_predicted_labels(y_pred_dict, dataset, split):
    dfs = []
    for var in y_pred_dict.keys():
        y = [x.item() for x in dataset.ann[var]]
        y_hat = [x.item() for x in y_pred_dict[var]]
        # map to labels if available (works for categorical variables)
        if var in dataset.label_mappings.keys():
            # Handle y_label with NaN checks correctly
            y = [dataset.label_mappings[var][int(x.item())] if not math.isnan(x.item()) else np.nan for x in dataset.ann[var]]
            y_hat = [dataset.label_mappings[var][int(x.item())] if not math.isnan(x.item()) else np.nan for x in y_pred_dict[var]]
        df = pd.DataFrame({
            'sample_id': dataset.samples,
            'var': var,
            'y': y,
            'y_hat': y_hat,
            'split': split
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# evaluate performance of off-the-shelf methods such as Random Forests and SVMs on regression/classification tasks
def evaluate_baseline_performance(train_dataset, test_dataset, variable_name, n_folds=5, n_jobs = 4):
    def prepare_data(data_object):
        # Concatenate Data Matrices
        X = np.concatenate([tensor for tensor in data_object.dat.values()], axis=1)

        # Prepare Labels
        y = np.array(data_object.ann[variable_name])

        # Filter out samples without a valid label
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        return X, y

    # Determine variable type
    variable_type = train_dataset.variable_types[variable_name]

    # Initialize models and parameter grids
    if variable_type == 'categorical':
        model_params = {
            'RandomForestClassifier': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None]
                }
            },
            'SVC': {
                'model': SVC(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly']
                }
            }
        }
    elif variable_type == 'numerical':
        model_params = {
            'RandomForestRegressor': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly']
                }
            }
        }

    # Cross-Validation and Training
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_train, y_train = prepare_data(train_dataset)
    print("Train:",X_train.shape)
    X_test, y_test = prepare_data(test_dataset)
    print("Test:",X_test.shape)

    metrics_list = []
    for model_name, mp in model_params.items():
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=kf, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict on test data
        y_pred = best_model.predict(X_test)

        # Evaluate predictions
        if variable_type == 'categorical':
            metrics = evaluate_classifier(y_test, y_pred)
        elif variable_type == 'numerical':
            metrics = evaluate_regressor(y_test, y_pred)

        for metric, value in metrics.items():
            metrics_list.append({
                'method': model_name,
                'var': variable_name,
                'variable_type': variable_type,
                'metric': metric,
                'value': value
            })

    # Convert the list of metrics to a DataFrame
    return pd.DataFrame(metrics_list)

def evaluate_baseline_survival_performance(train_dataset, test_dataset, duration_col, event_col, n_folds=5, n_jobs=4):
    print(f"[INFO] Evaluating baseline survival prediction performance")
    def prepare_data(data_object, duration_col, event_col):
        # Concatenate Data Matrices
        X = np.concatenate([tensor for tensor in data_object.dat.values()], axis=1)
        
        # Prepare Survival Data (Durations and Events)
        durations = np.array(data_object.ann[duration_col])
        events = np.array(data_object.ann[event_col])
        y = np.array([(event, duration) for event, duration in zip(events, durations)], 
                     dtype=[('Event', '?'), ('Time', '<f8')])
        
        # Filter out samples without a valid survival data
        valid_indices = ~np.isnan(durations) & ~np.isnan(events)
        X = X[valid_indices]
        y = y[valid_indices]
        return X, y

    # Prepare train and test data
    X_train, y_train = prepare_data(train_dataset, duration_col, event_col)
    X_test, y_test = prepare_data(test_dataset, duration_col, event_col)

    # Initialize Random Survival Forest
    rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, min_samples_split=10,
                               min_samples_leaf=15, max_features="sqrt", n_jobs=n_jobs, random_state=42)

    # Cross-Validation to determine the best model
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    c_index_scores = []

    for train_index, test_index in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
        
        rsf.fit(X_fold_train, y_fold_train)
        prediction = rsf.predict(X_fold_test)
        c_index = concordance_index_censored(y_fold_test['Event'], y_fold_test['Time'], prediction)
        c_index_scores.append(c_index[0])

    # Calculate average C-index across all folds
    avg_c_index = np.mean(c_index_scores)
    print(f"[INFO] Average C-index in cross-validation: {avg_c_index}")

    # Retrain on full training data and evaluate on test data
    rsf.fit(X_train, y_train)
    test_prediction = rsf.predict(X_test)
    test_c_index = concordance_index_censored(y_test['Event'], y_test['Time'], test_prediction)
    print(f"[INFO] C-index on test data: {test_c_index[0]}")

    # Reporting
    metrics_list = [{
        'method': 'RandomSurvivalForest',
        'var': event_col, 
        'variable_type': 'numerical',
        'metric': 'cindex',
        'value': test_c_index[0]
    }]
    
    return pd.DataFrame(metrics_list)

def remove_batch_associated_variables(data, variable_types, target_dict, batch_dict = None, mi_threshold=0.1):
    """
    Filter the data matrix to keep only the columns that are predictive of the target variables 
    and not predictive of the batch variables.
    
    Args:
        data (pd.DataFrame): The data matrix.
        target_dict (dict): A dictionary of target variables.
        batch_dict (dict): A dictionary of batch variables.
        variable_types (dict): A dictionary of variable types (either "numerical" or "categorical").
        mi_threshold (float, optional): The mutual information threshold for a column to be considered predictive.
                                        Defaults to 0.1.
    
    Returns:
        pd.DataFrame: The filtered data matrix.
    """
    # Convert target and batch tensors to numpy
    target_dict_np = {k: v.numpy() for k, v in target_dict.items()}

    important_features = set()

    # Find important features for target variables
    for var_name, target in target_dict_np.items():
        # Skip if all values are missing
        if np.all(np.isnan(target)):
            continue
            
        # Subset data and target where target is not missing
        not_missing = ~np.isnan(target)
        data_sub = data[not_missing]
        target_sub = target[not_missing]

        if variable_types[var_name] == "categorical":
            clf = RandomForestClassifier()
        else:  # numerical
            clf = RandomForestRegressor()
            
        clf = clf.fit(data_sub, target_sub)
        model = SelectFromModel(clf, prefit=True)
        important_features.update(data.columns[model.get_support()])

    if batch_dict is not None:
        batch_dict_np = {k: v.numpy() for k, v in batch_dict.items()}
        # Compute mutual information for batch variables
        for var_name, batch in batch_dict_np.items():
            # Skip if all values are missing
            if np.all(np.isnan(batch)):
                continue

            # Subset data and batch where batch is not missing
            not_missing = ~np.isnan(batch)
            data_sub = data[not_missing]
            batch_sub = batch[not_missing]

            if variable_types[var_name] == "categorical":
                mi = mutual_info_classif(data_sub, batch_sub)
            else:  # numerical
                mi = mutual_info_regression(data_sub, batch_sub)

            # Remove features with high mutual information with batch variables
            important_features -= set(data.columns[mi > mi_threshold])

    return data[list(important_features)]


def get_important_features(model, var, top=20):
    # Ensure that the variable exists in the model's feature importances
    if var not in model.feature_importances:
        print(f"No feature importances found for variable: {var}")
        return None

    # Fetch the dataframe for the specified variable
    df_imp = model.feature_importances[var]

    top_features = df_imp.groupby(['target_class']).apply(lambda x: x.nlargest(top, 'importance')).reset_index(drop=True)

    return top_features

def subset_assays_by_features(dataset, features_dict):
    # Find indices of the features in the corresponding 
    # data matrix for each key in features_dict
    subset_dat = {}
    for layer in features_dict.keys():
        indices = [dataset.features[layer].get_loc(x) for x in features_dict[layer]]
        subset_dat[layer] = dataset.dat[layer][:, indices]
    # Convert subset_dat to pandas DataFrame and prepend feature names with layer names
    df_list = []
    for layer, data in subset_dat.items():
        # Convert matrix to DataFrame
        df_temp = pd.DataFrame(data)
        
        # Rename columns to prepend with layer name
        df_temp.columns = [f"{layer}_{feature}" for feature in features_dict[layer]]
        df_list.append(df_temp)
    # Concatenate dataframes horizontally
    concatenated_df = pd.concat(df_list, axis=1)
    return concatenated_df    

# Accepts as input a MultiOmicDataset object and prints summary stats per variable
def print_summary_stats(dataset):
    for var, tensor in dataset.ann.items():
        print(f"Summary for variable: {var}")
        
        if dataset.variable_types[var] == "categorical":
            # Handle Categorical Variable
            unique_vals, counts = np.unique(tensor, return_counts=True)
            print("Categorical Variable Summary:")
            
            for uv, cnt in zip(unique_vals, counts):
                original_label = dataset.label_mappings.get(var, {}).get(uv, uv)  # Fall back to uv if mapping doesn't exist
                print(f"  Label: {original_label}, Count: {cnt}")

        elif dataset.variable_types[var] == "numerical":
            # Handle Numerical Variable
            median_val = np.nanmedian(tensor)
            mean_val = np.nanmean(tensor)
            print(f"Numerical Variable Summary: Median = {median_val}, Mean = {mean_val}")
        print("------")
        

def plot_kaplan_meier_curves(durations, events, categorical_variable):
    """
    Plots Kaplan-Meier survival curves for different groups defined by a categorical variable.

    Parameters:
    - durations: An array-like object of survival times or durations.
    - events: An array-like object indicating whether an event (e.g., death) occurred (1) or was censored (0).
    - categorical_variable: An array-like object defining groups for plotting different survival curves.
    """
    # Initialize the Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Ensure data is in a pandas DataFrame for easy handling
    data = pd.DataFrame({
        'Duration': durations,
        'Event': events,
        'Group': categorical_variable
    })
    
    # Plot survival curves for each category
    plt.figure(figsize=(10, 6))
    categories = data['Group'].unique()
    for category in categories:
        # Select data for the group
        group_data = data[data['Group'] == category]
        
        # Fit the model
        kmf.fit(durations=group_data['Duration'], event_observed=group_data['Event'], label=str(category))
        
        # Plot the survival curve for the group
        kmf.plot_survival_function()
    
    plt.title('Kaplan-Meier Survival Curves by Group')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend(title='Group')
    plt.grid(True)
    plt.show()
    

def plot_hazard_ratios(cox_model):
    """
    Plots the sorted log hazard ratios from a fitted Cox Proportional Hazards model,
    sorted by their p-values and annotated with stars to indicate levels of
    statistical significance.

    Parameters:
    - cox_model: A fitted CoxPH model from the lifelines package.
    """
    # Extract the coefficients (log hazard ratios), their confidence intervals, and p-values
    coef_summary = cox_model.summary[['coef', 'coef lower 95%', 'coef upper 95%', 'p']]
    
    # Sort the DataFrame by the p-values
    coef_summary_sorted = coef_summary.sort_values(by='p')
    
    # Calculate the error bars
    errors = np.abs(coef_summary_sorted[['coef lower 95%', 'coef upper 95%']].subtract(coef_summary_sorted['coef'], axis=0).values.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(coef_summary_sorted))
    ax.errorbar(coef_summary_sorted['coef'], y_positions, xerr=errors, fmt='o', color='skyblue', capsize=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_summary_sorted.index)
    ax.axvline(x=0, color='grey', linestyle='--')
    
    p_thresholds = [(0.0001, '***'), (0.001, '**'), (0.05, '*'), (0.1, '.')]
    annotations = ['' if p > 0.05 else next(stars for threshold, stars in p_thresholds if p < threshold) 
                   for p in coef_summary_sorted['p']]

    for i, annotation in enumerate(annotations):
        ax.text(coef_summary_sorted['coef'][i], y_positions[i], f'  {annotation}', verticalalignment='center')
    
    ax.invert_yaxis()  # Invert y-axis so the most significant variables are at the top
    plt.xlabel('Log Hazard Ratio')
    plt.title('Log Hazard Ratios Sorted by P-Value with 95% CI')
    plt.grid(axis='x', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.show()
    

def build_cox_model(df, duration_col, event_col):
    """
    Fits a Cox Proportional Hazards model to the data.

    Parameters:
    - df: Pandas DataFrame containing the clinical variables, predicted risk scores,
          durations, and event indicators.
    - duration_col: The name of the column in df that contains the survival times.
    - event_col: The name of the column in df that contains the event occurrence indicator (1 if event occurred, 0 otherwise).

    Returns:
    - cox_model: Fitted CoxPH model.
    """
    
    def remove_low_variance_survival_features(df, duration_col, event_col, threshold=0.01):
        events = df[event_col].astype(bool)
        low_variance_features = []

        for feature in df.drop(columns=[duration_col, event_col]).columns:
            # Calculate variance within each group (event occurred and not occurred)
            variance_when_event = df.loc[events, feature].var()
            variance_when_no_event = df.loc[~events, feature].var()

            # If variance in both groups is below the threshold, mark for removal
            if variance_when_event < threshold or variance_when_no_event < threshold:
                low_variance_features.append(feature)
    
        # Remove identified low variance features
        df_filtered = df.drop(columns=low_variance_features)
        # Report removed features
        if low_variance_features:
            print("Removed low variance features due to conditioning on event:", low_variance_features)
        else:
            print("No low variance features were removed based on event conditioning.")
        return df_filtered    
    
    # remove uninformative features 
    df = remove_low_variance_survival_features(df, duration_col, event_col)
                                      
    # Initialize the Cox Proportional Hazards model
    cox_model = CoxPHFitter()

    # Fit the model
    cox_model.fit(df, duration_col=duration_col, event_col=event_col)

    return cox_model


def k_means_clustering(data, k):
    """
    Perform k-means clustering on a given pandas DataFrame.

    Parameters:
    - data: pandas DataFrame, where rows are samples and columns are features.
    - k: int, the number of clusters to form.

    Returns:
    - cluster_labels: A pandas Series indicating the cluster label for each sample.
    - kmeans: The fitted KMeans instance, which can be used to access cluster centers and other attributes.
    """
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)

    # Fit the model to the data
    kmeans.fit(data)

    # Extract the cluster labels for each data point
    cluster_labels = pd.Series(kmeans.labels_, index=data.index)

    return cluster_labels, kmeans

def louvain_clustering(X, threshold=None, k=None):
    """
    Create a graph from pairwise distances within X. You can define a threshold to connect edges
    or specify k for k-nearest neighbors.
    
    Parameters:
    - X: numpy array, shape (n_samples, n_features)
    - threshold: float, distance threshold to create an edge between two nodes.
    - k: int, number of nearest neighbors to connect for each node.
    
    Returns:
    - G: a networkx graph
    """
    distances = euclidean_distances(X)
    G = nx.Graph()
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[0]):
            # If a threshold is defined, use it to create edges
            if threshold is not None and distances[i, j] < threshold:
                G.add_edge(i, j, weight=1/distances[i, j])
            # If k is defined, add an edge if j is one of i's k-nearest neighbors
            elif k is not None:
                if np.argsort(distances[i])[:k + 1].__contains__(j):
                    G.add_edge(i, j, weight=1/distances[i, j])
    partition = community_louvain.best_partition(G)
    
    cluster_labels = np.full(len(X), np.nan, dtype=float)
    # Fill the array with the cluster labels from the partition dictionary
    for node_id, cluster_label in partition.items():
        if node_id in range(len(X)):  # Check if the node_id is a valid index in X
            cluster_labels[node_id] = cluster_label
        else:
            # If node_id is not a valid index in X, it's already set to NaN
            continue

    return cluster_labels, G, partition


def get_optimal_clusters(data, min_k=2, max_k=10):
    """
    Find the optimal number of clusters (k) for k-means clustering on the given data,
    based on the silhouette score, and return the cluster labels for the optimal k.

    Parameters:
    - data: pandas DataFrame or numpy array, dataset for clustering.
    - min_k: int, minimum number of clusters to try.
    - max_k: int, maximum number of clusters to try.

    Returns:
    - int, the optimal number of clusters based on the silhouette score.
    - DataFrame, silhouette scores for each k.
    - array, cluster labels for the optimal number of clusters.
    """
    silhouette_scores = []
    cluster_labels_dict = {}  # To store cluster labels for each k

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init = 'auto', random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append((k, silhouette_avg))
        cluster_labels_dict[k] = cluster_labels  # Store cluster labels
        #print(f"Number of clusters: {k}, Silhouette Score: {silhouette_avg:.4f}")
    
    # Convert silhouette scores to DataFrame for easier handling and visualization
    silhouette_scores_df = pd.DataFrame(silhouette_scores, columns=['k', 'silhouette_score'])
    
    # Find the optimal k (number of clusters) with the highest silhouette score
    optimal_k = silhouette_scores_df.loc[silhouette_scores_df['silhouette_score'].idxmax()]['k']
    
    # Retrieve the cluster labels for the optimal k
    optimal_cluster_labels = cluster_labels_dict[optimal_k]
    
    return optimal_cluster_labels, optimal_k, silhouette_scores_df

# compute adjusted rand index; adjusted mutual information for two sets of paired labels
def compute_ami_ari(labels1, labels2):
    def convert_nan (labels):
        return ['unavailable' if pd.isna(x) else x for x in labels]
    labels1 = convert_nan(labels1)
    labels2 = convert_nan(labels2)
    ami = adjusted_mutual_info_score(labels1, labels2)
    ari = adjusted_rand_score(labels1, labels2)
    return {'ami': ami, 'ari': ari}


def plot_label_concordance_heatmap(labels1, labels2, figsize=(12, 10)):
    """
    Plot a heatmap reflecting the concordance between two sets of labels using pandas crosstab.

    Parameters:
    - labels1: The first set of labels.
    - labels2: The second set of labels.
    """
    # Compute the cross-tabulation
    ct = pd.crosstab(pd.Series(labels1, name='Labels Set 1'), pd.Series(labels2, name='Labels Set 2'))
    # Normalize the cross-tabulation matrix column-wise
    ct_normalized = ct.div(ct.sum(axis=1), axis=0)
    
    # Plot the heatmap
    plt.figure(figsize = figsize)
    sns.heatmap(ct_normalized, annot=True,cmap='viridis', linewidths=.5)# col_cluster=False)
    plt.title('Concordance between label groups')
    plt.show()
    

class CBioPortalData:
    def __init__(self, base_url=None, study_id=None, data_files=None):
        self.base_url = base_url if base_url is not None else "https://cbioportal-datahub.s3.amazonaws.com"
        self.study_id = study_id
        self.data_files = data_files if data_files is not None else []
        self.data_tables = {}  # Initialize data_tables as an empty dictionary
        
        logging.basicConfig(level=logging.INFO, format='[INFO] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CBioPortalData")
        archive_path = self.download_study_archive()
        study_dir = self.extract_archive(archive_path)
        self.read_data()
        self.list_data_files()

    def download_study_archive(self):
        url = os.path.join(self.base_url, f"{self.study_id}.tar.gz")
        dest_file = f"{self.study_id}.tar.gz"
        if not os.path.exists(dest_file):
            self.logger.info(f"Downloading {self.study_id} data archive")
            r = requests.get(url)
            with open(dest_file, 'wb') as f:
                f.write(r.content)
        return dest_file

    def extract_archive(self, archive_path):
        self.logger.info(f"Extracting {archive_path}")
        base = archive_path.split('.')[0]
        if not os.path.exists(base):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall()
        self.data_files = [f for f in glob(os.path.join(base, "data_*.txt"))]
        return base

    def read_data(self, files=None):
        if files is None:
            files = self.data_files
        
        self.data_tables = {}
        for file_path in tqdm(files, desc="Processing Files"):
            file_name = os.path.basename(file_path)
            self.logger.info(f"Importing data file: {file_name}")
            df = pd.read_csv(file_path, comment='#', sep='\t', low_memory=False)

            if re.search('mutations', file_name):
                self.logger.info(f"Binarizing and converting to matrix: {file_name}")
                df = self.binarize_mutations(df)
            elif not re.search('clinical|drug_treatment', file_name) and "Hugo_Symbol" in df.columns:
                self.logger.info(f"Converting {file_name} to matrix")
                df = self.process_data(df)

            clean_name = re.sub(r"data_|\.txt", "", file_name)
            self.data_tables[clean_name] = df

    def process_data(self, df):
        # Exclude 'Hugo_Symbol' and 'Entrez_Gene_Id' fields
        cols = [col for col in df.columns if col not in ('Hugo_Symbol', 'Entrez_Gene_Id')]
        # Remove non-unique rows based on 'Hugo_Symbol'; keep first among duplicates. 
        df_unique = df.drop_duplicates(subset=['Hugo_Symbol'], keep='first')
        # Set 'Hugo_Symbol' as index and select the desired columns
        df_processed = df_unique.set_index('Hugo_Symbol')[cols]
        return df_processed

    def binarize_mutations(self, df):
        # Check if required columns exist
        if 'Hugo_Symbol' not in df.columns or 'Tumor_Sample_Barcode' not in df.columns:
            raise ValueError("Required columns are missing.")

        # Group by 'Hugo_Symbol' and 'Tumor_Sample_Barcode', count mutations
        mutation_counts = df.groupby(['Hugo_Symbol', 'Tumor_Sample_Barcode']).size().reset_index(name='counts')

        # Pivot table to create a matrix (genes vs samples)
        df_pivot = mutation_counts.pivot(index='Hugo_Symbol', columns='Tumor_Sample_Barcode', values='counts').fillna(0)

        # Binarize the matrix
        df_binary = (df_pivot > 0).astype(int)

        # Process the binary data further if needed, similar to process_data method
        df_processed = self.process_data(df_binary.reset_index())

        return df_processed
    
    def list_data_files(self):
        # Create a DataFrame with file names
        return {x: self.data_tables[x].shape for x in self.data_tables.keys()}

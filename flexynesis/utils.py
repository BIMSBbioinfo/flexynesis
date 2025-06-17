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
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.utils import resample

from scipy.stats import pearsonr, linregress


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from xgboost import XGBClassifier, XGBRegressor

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from plotnine import (
    ggplot, aes, geom_point, geom_smooth, geom_line, geom_abline, geom_step,
    labs, ggtitle, annotate, theme_minimal, theme, element_text,
    scale_color_manual, scale_color_gradient, scale_color_brewer,
    geom_errorbarh, geom_text,
    theme_bw, theme, element_blank, scale_y_discrete
)
    
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import community as community_louvain

from sklearn.preprocessing import StandardScaler
import ot 


def plot_dim_reduced(matrix, labels, method='pca', color_type='categorical', title=None):
    """
    Plots the first two dimensions of the transformed input matrix using plotnine,
    with PCA or UMAP, and includes explained variance for PCA.

    Args:
        matrix (np.array): Input data matrix (n_samples, n_features).
        labels (list or array): Labels for coloring.
        method (str): 'pca' or 'umap'.
        color_type (str): 'categorical' or 'numerical'.
        title (str or None): Optional title for the plot.
    """
    method = method.lower()

    # Fit transformation
    if method == 'pca':
        transformer = PCA(n_components=2)
        transformed_matrix = transformer.fit_transform(matrix)
        var_exp = transformer.explained_variance_ratio_ * 100
        xlab = f"PC1 ({var_exp[0]:.1f}%)"
        ylab = f"PC2 ({var_exp[1]:.1f}%)"
        colnames = ['PC1', 'PC2']
    elif method == 'umap':
        transformer = UMAP(n_components=2)
        transformed_matrix = transformer.fit_transform(matrix)
        xlab = "UMAP1"
        ylab = "UMAP2"
        colnames = ['UMAP1', 'UMAP2']
    else:
        raise ValueError("Invalid method. Expected 'pca' or 'umap'.")

    # Create DataFrame
    df = pd.DataFrame(transformed_matrix, columns=colnames)
    df["Label"] = list(labels)

    # Title
    plot_title = title if title else f"{method.upper()} Scatter Plot"

    # Plot
    if color_type == 'categorical':
        df["Label"] = df["Label"].astype(str)
        plot = (
            ggplot(df, aes(x=colnames[0], y=colnames[1], color='Label')) +
            geom_point() +
            labs(title=plot_title, x=xlab, y=ylab, color="Labels") +
            theme_minimal()
        )
    elif color_type == 'numerical':
        df["Label"] = pd.to_numeric(df["Label"], errors='coerce')
        plot = (
            ggplot(df, aes(x=colnames[0], y=colnames[1], color='Label')) +
            geom_point() +
            scale_color_gradient(low="blue", high="red") +
            labs(title=plot_title, x=xlab, y=ylab, color="Label") +
            theme_minimal()
        )
    else:
        raise ValueError("Invalid color_type. Choose 'categorical' or 'numerical'.")

    return plot



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
    
    # Create DataFrame
    df = pd.DataFrame({"True Values": true_values, "Predicted Values": predicted_values})
    
    # Generate scatter plot with regression line
    plot = (
        ggplot(df, aes(x="True Values", y="Predicted Values")) +
        geom_point(alpha=0.5) +
        geom_smooth(method='lm', color='red') +
        annotate("text", x=min(true_values), y=max(predicted_values), label=corr_text, ha='left', va='top', size=10) +
        labs(
            title="True vs Predicted Values",
            x="True Values",
            y="Predicted Values"
        ) +
        theme_minimal()
    )
    
    return plot
    
    
from scipy.stats import mannwhitneyu, kruskal

def plot_boxplot(categorical_x, numerical_y, title_x='Categories', title_y='Values', figsize=(10, 6), jittersize = 4):
    df = pd.DataFrame({title_x: categorical_x, title_y: numerical_y})
    
    # Compute p-value
    groups = df[title_x].unique()
    if len(groups) == 2:
        group1 = df[df[title_x] == groups[0]][title_y]
        group2 = df[df[title_x] == groups[1]][title_y]
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U"
    else:
        group_data = [df[df[title_x] == group][title_y] for group in groups]
        stat, p = kruskal(*group_data)
        test_name = "Kruskal-Wallis"

    # Create a boxplot with jittered points
    plt.figure(figsize=figsize)
    sns.boxplot(x=title_x, y=title_y,  hue=title_x, data=df, palette='Set2', legend=False, fill= False)
    sns.stripplot(x=title_x, y=title_y, data=df, color='black', size=jittersize, jitter=True, dodge=True, alpha=0.4)

    # Labels and p-value annotation
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.text(
        x=-0.4,
        y=plt.ylim()[1],
        s=f'{test_name} p = {p:.3e}',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
    )

    plt.tight_layout()
    plt.show()
    
# given a vector of numerical values which may contain 
# NAN values, return a binary grouping based on median values 
def split_by_median(v):
    return ((v - torch.nanmedian(v)) > 0).float()
    
def evaluate_survival(outputs, durations, events):
    """
    Computes the concordance index (c-index) for survival predictions.

    Parameters:
    - outputs: Predicted risk scores (torch.Tensor or np.array).
    - durations: True survival times or durations (torch.Tensor or np.array).
    - events: Event indicators (1=event, 0=censored) (torch.Tensor or np.array).

    Returns:
    - A dictionary containing the c-index.
    """
    # Ensure all inputs are NumPy arrays
    if isinstance(durations, torch.Tensor):
        durations = durations.numpy()
    if isinstance(events, torch.Tensor):
        events = events.numpy()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()

    # Remove entries with NaNs
    valid_mask = ~np.isnan(durations) & ~np.isnan(events)
    if valid_mask.sum() > 0:
        durations = durations[valid_mask]
        events = events[valid_mask]
        outputs = outputs[valid_mask]

    # Compute concordance index (lifelines expects higher risk → lower survival)
    c_index = concordance_index(durations, -outputs, events)
    return {'cindex': c_index}

def generate_bootstrap_indices(n, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=n, replace=True) for _ in range(n_bootstraps)]

# bootstrapping function for regression/classification tasks 
def bootstrap_metric(y_true, y_pred, indices_list, metric_fn, ci = 95, **kwargs):
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for idx in indices_list:
        score = metric_fn(y_true[idx], y_pred[idx], **kwargs)
        scores.append(score)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return scores, (np.mean(scores), lower, upper)

def evaluate_classifier(y_true, y_probs, print_report=False):
    """
    Evaluate the performance of a classifier using multiple metrics and optionally print a detailed classification report.

    This function computes balanced accuracy, F1 score (weighted), Cohen's Kappa score, average AUROC score, and 
    weighted-average AUC-PR score for the given true labels and predicted probabilities.
    If `print_report` is set to True, it prints a detailed classification report.

    Args:
        y_true (array-like): True labels of the data, must be a 1D list or array of labels.
        y_probs (array-like): Predicted probabilities for each class, must be 2D (n_samples, n_classes).
        print_report (bool, optional): If True, prints a detailed classification report. Defaults to False.

    Returns:
        dict: A dictionary containing:
              - 'balanced_acc': The balanced accuracy of the predictions.
              - 'f1_score': The weighted-average F1 score of the predictions.
              - 'kappa': Cohen's Kappa score indicating the level of agreement between the true and predicted labels.
              - 'average_auroc': The weighted average AUROC score across all classes.
              - 'average_aupr': The weighted average AUC-PR score across all classes.
    """
    # Convert probabilities to predicted labels
    y_pred = np.argmax(y_probs, axis=1)

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # F1 score (weighted)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Compute AUROC/AUPR 
    try:
        if y_probs.shape[1] == 2:  # Binary classification
            y_probs_binary = y_probs[:, 1]  # Use positive class probabilities
            average_auroc = roc_auc_score(y_true, y_probs_binary)
            average_aupr = average_precision_score(y_true, y_probs_binary)  # AUC-PR for binary case
        else:  # Multiclass classification
            average_auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
            average_aupr = average_precision_score(y_true, y_probs, average='weighted')  # Weighted AUC-PR for multiclass
    except ValueError:
        average_auroc = None  # Handle cases where AUROC cannot be computed
        average_aupr = None  # Handle cases where AUC-PR cannot be computed

    # Full classification report
    if print_report:
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)

    return {
        "balanced_acc": balanced_acc,
        "f1_score": f1,
        "kappa": kappa,
        "average_auroc": average_auroc,
        "average_aupr": average_aupr  # Added AUC-PR
    }

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_roc_curves(y_true, y_probs):
    """
    Plot ROC curves using plotnine for binary or multiclass classification.

    Args:
        y_true (array-like): True class labels.
        y_probs (array-like): Predicted probabilities (n_samples, n_classes).
    """
    y_true = np.array(y_true)
    n_classes = y_probs.shape[1]
    plot_data = []

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        auc_score = roc_auc_score(y_true, y_probs[:, 1])
        plot_data.append(pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'label': [f'Class 1 (AUC = {auc_score:.2f})'] * len(fpr)}))
    else:
        # Multiclass classification
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        for i in classes:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'label': [f'Class {i} (AUC = {auc_score:.2f})'] * len(fpr)
            })
            plot_data.append(df)

    # Combine all data
    all_data = pd.concat(plot_data, ignore_index=True)

    # Plot using plotnine
    roc_plot = (
        ggplot(all_data, aes(x='fpr', y='tpr', color='label')) +
        geom_line(size=1.2) +
        geom_abline(intercept=0, slope=1, linetype='dashed', color='gray') +
        labs(
            title='ROC Curve',
            x='False Positive Rate',
            y='True Positive Rate'
        ) +
        theme_minimal()
    )

    return roc_plot

def plot_pr_curves(y_true, y_probs):
    """
    Plot Precision-Recall (PR) curves using plotnine for binary or multiclass classification.

    Args:
        y_true (array-like): True class labels.
        y_probs (array-like): Predicted probabilities (n_samples, n_classes).
    """
    y_true = np.array(y_true)
    n_classes = y_probs.shape[1]
    plot_data = []

    if n_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        aupr = average_precision_score(y_true, y_probs[:, 1])
        plot_data.append(pd.DataFrame({
            'recall': recall,
            'precision': precision,
            'label': [f'Class 1 (AUPR = {aupr:.2f})'] * len(recall)
        }))
    else:
        # Multiclass classification (one-vs-rest)
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        for i in classes:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            aupr = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            df = pd.DataFrame({
                'recall': recall,
                'precision': precision,
                'label': [f'Class {i} (AUPR = {aupr:.2f})'] * len(recall)
            })
            plot_data.append(df)

    # Combine all data
    all_data = pd.concat(plot_data, ignore_index=True)

    # Plot using plotnine
    pr_plot = (
        ggplot(all_data, aes(x='recall', y='precision', color='label')) +
        geom_line(size=1.2) +
        labs(
            title='Precision-Recall Curve',
            x='Recall',
            y='Precision'
        ) +
        theme_minimal()
    )

    return pr_plot


def evaluate_regressor(y_true, y_pred):
    """
    Evaluate the performance of a regression model using mean squared error, R-squared, and Pearson correlation coefficient.

    This function computes the mean squared error (MSE) between true and predicted values as a measure of prediction accuracy.
    It also performs a linear regression analysis between the true and predicted values to obtain the R-squared value, which
    explains the variance ratio, and the Pearson correlation coefficient, providing insight into the linear relationship strength.

    Args:
        y_true (array-like): True values of the dependent variable, must be a 1D list or array.
        y_pred (array-like): Predicted values as returned by a regressor, must match the dimensions of y_true.

    Returns:
        dict: A dictionary containing:
              - 'mse': The mean squared error between the true and predicted values.
              - 'r2': The R-squared value indicating the proportion of variance in the dependent variable predictable from the independent variable.
              - 'pearson_corr': The Pearson correlation coefficient indicating the linear relationship strength between the true and predicted values.
    """
    mse = mean_squared_error(y_true, y_pred)
    slope, intercept, r_value, p_value, std_err = linregress(y_true,y_pred)
    r2 = r_value**2 
    return {"mse": mse, "r2": r2, "pearson_corr": r_value}

def evaluate_wrapper(method, y_pred_dict, dataset, surv_event_var = None, surv_time_var = None):
    """
    Evaluates predictions for different variables within a dataset using appropriate metrics based on the variable type. 
    Supports evaluation for numerical, categorical, and survival data.

    This function loops through each variable in the predictions dictionary, determines the type of the variable,
    and evaluates the predictions using the appropriate method: regression, classification, or survival analysis.
    It compiles the metrics into a list of dictionaries, which is then converted into a pandas DataFrame.

    Args:
        method (str): Identifier for the prediction method or model used.
        y_pred_dict (dict): A dictionary where keys are variable names and values are arrays of predicted values.
        dataset (Dataset): A dataset object containing actual values and metadata such as variable types.
        surv_event_var (str, optional): The name of the survival event variable. Required if survival analysis is performed.
        surv_time_var (str, optional): The name of the survival time variable. Required if survival analysis is performed.

    Returns:
        pd.DataFrame: A DataFrame where each row contains the method, variable name, variable type, metric name, and metric value.

    """
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
                'method': method,
                'var': var,
                'variable_type': dataset.variable_types[var],
                'metric': metric,
                'value': value
            })
    # Convert the list of metrics to a DataFrame
    return pd.DataFrame(metrics_list)

def get_predicted_labels(y_pred_dict, dataset, split, method_name):
    """
    Generate a DataFrame with class probabilities and associated metadata.

    Args:
        y_pred_dict (dict): Dictionary containing predicted probabilities for each variable.
        dataset: Dataset object containing variable types, label mappings, and sample information.
        split (str): Split identifier (e.g., 'train', 'test', or 'val').
        method_name: Name of the method used for prediction task

    Returns:
        pd.DataFrame: A DataFrame containing:
            - sample_id
            - variable
            - class label
            - probability for that class label
            - known label (y_true)
            - predicted label (argmax of the probabilities)
            - train/test split
            - method_name 
    """
    dfs = []

    for var in y_pred_dict.keys():
        if dataset.variable_types[var] == 'categorical':
            # Predicted probabilities
            probabilities = y_pred_dict[var]

            # Convert class indices to labels if mappings exist
            if var in dataset.label_mappings.keys():
                class_labels = [dataset.label_mappings[var][idx] for idx in range(probabilities.shape[1])]
            else:
                class_labels = [f'class_{i}' for i in range(probabilities.shape[1])]

            # Get true labels (y_true)
            y_true = [dataset.label_mappings[var][int(x.item())] if var in dataset.label_mappings.keys() and not np.isnan(x.item()) else np.nan for x in dataset.ann[var]]

            # Predicted labels (argmax of probabilities)
            y_pred_indices = np.argmax(probabilities, axis=1)
            y_pred = [dataset.label_mappings[var][idx] if var in dataset.label_mappings.keys() else idx for idx in y_pred_indices]

            # Create a DataFrame for each sample and its probabilities
            for i, sample_id in enumerate(dataset.samples):
                for j, class_label in enumerate(class_labels):
                    dfs.append({
                        'sample_id': sample_id,
                        'variable': var,
                        'class_label': class_label,
                        'probability': probabilities[i, j],
                        'known_label': y_true[i],
                        'predicted_label': y_pred[i],
                        'split': split,
                        'method': method_name
                    })
        else:
            # For numerical variables, set class_label and probability to NA
            y_true = [x.item() for x in dataset.ann[var]]
            y_pred = [x.item() for x in y_pred_dict[var]]
            for i, sample_id in enumerate(dataset.samples):
                dfs.append({
                    'sample_id': sample_id,
                    'variable': var,
                    'class_label': np.nan,
                    'probability': np.nan,
                    'known_label': y_true[i],
                    'predicted_label': y_pred[i],
                    'split': split,
                    'method': method_name
                })

    # Combine all rows into a DataFrame
    return pd.DataFrame(dfs)




def evaluate_baseline_performance(train_dataset, test_dataset, variable_name, methods, n_folds=5, n_jobs=4, use_pca=False, n_components=100):
    """
    Evaluates the performance of machine learning models on a given variable with optional PCA for dimensionality reduction.

    Args:
        train_dataset (Dataset): A MultiOmicDataset object containing training data and metadata.
        test_dataset (Dataset): A MultiOmicDataset object containing testing data.
        variable_name (str): The name of the target variable for prediction.
        methods (list of str): List of machine learning methods to evaluate, e.g., ['RandomForest', 'SVM', 'XGBoost'].
        n_folds (int, optional): Number of folds for K-fold cross-validation. Defaults to 5.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 4.
        use_pca (bool, optional): Whether to apply PCA for dimensionality reduction. Defaults to False.
        n_components (int, optional): Number of principal components to keep if PCA is applied. Defaults to 50.

    Returns:
        pd.DataFrame: A DataFrame containing metrics for each method.
    """
    def prepare_data(data_object, pca_model=None, fit_pca=False):
        # Concatenate Data Matrices
        X = np.concatenate([tensor for tensor in data_object.dat.values()], axis=1)
        y = np.array(data_object.ann[variable_name])

        # Filter out samples without a valid label
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]

        if use_pca:
            if fit_pca:
                pca_model.fit(X)
                print(f"PCA fitted: Reduced to {n_components} components")
            X = pca_model.transform(X)
            print(f"PCA applied: Transformed to {n_components} components")

        return X, y, np.where(valid_indices)[0]

    # Initialize PCA model if PCA is used
    pca_model = PCA(n_components=n_components) if use_pca else None

    # Determine variable type
    variable_type = train_dataset.variable_types[variable_name]

    # Cross-Validation and Training
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_train, y_train, train_indices = prepare_data(train_dataset, pca_model=pca_model, fit_pca=True)
    print("Train:", X_train.shape)
    X_test, y_test, test_indices = prepare_data(test_dataset, pca_model=pca_model, fit_pca=False)
    print("Test:", X_test.shape)

    metrics_list = []
    predictions = []  # Collect all predictions
    
    for method in methods:
        if variable_type == 'categorical':
            if method == 'RandomForest':
                model = RandomForestClassifier(random_state=42)
                params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
            elif method == 'SVM':
                model = SVC(probability=True, random_state=42)
                params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly']}
            elif method == 'XGBoost':
                model = XGBClassifier(eval_metric='logloss', random_state=42)
                params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}
        elif variable_type == 'numerical':
            if method == 'RandomForest':
                model = RandomForestRegressor(random_state=42)
                params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
            elif method == 'SVM':
                model = SVR()
                params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly']}
            elif method == 'XGBoost':
                model = XGBRegressor(random_state=42)
                params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}

        print("Training method:", method)
        grid_search = GridSearchCV(model, params, cv=kf, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict on test data
        if variable_type == 'categorical':
            y_probs = best_model.predict_proba(X_test)
            metrics = evaluate_classifier(y_test, y_probs, print_report=True)
            y_pred_dict = {variable_name: y_probs}
        elif variable_type == 'numerical':
            y_pred = best_model.predict(X_test)
            metrics = evaluate_regressor(y_test, y_pred)
            y_pred_dict = {variable_name: y_pred}

        # need to get test indices to only consider samples with labels
        df_preds = get_predicted_labels(y_pred_dict, test_dataset.subset(test_indices), 'test', method)
        predictions.append(df_preds)
            
        for metric, value in metrics.items():
            metrics_list.append({
                'method': method + ('Classifier' if variable_type == 'categorical' else 'Regressor'),
                'var': variable_name,
                'variable_type': variable_type,
                'metric': metric,
                'value': value
            })

    predictions = pd.concat(predictions, ignore_index=True)
    
    return pd.DataFrame(metrics_list), predictions



def evaluate_baseline_survival_performance(train_dataset, test_dataset, duration_col, event_col, n_folds=5, n_jobs=4):
    """
    Evaluates the baseline performance of a Random Survival Forest model on survival data using the Concordance Index.

    The function preprocesses both training and testing datasets to prepare appropriate survival data (comprising durations 
    and event occurrences), performs cross-validation to assess model robustness, and then calculates the Concordance Index on 
    the test data. It uses a Random Survival Forest (RSF) as the predictive model.

    Args:
        train_dataset (Dataset): The training dataset (a MultiOmicDataset object) containing features and survival data.
        test_dataset (Dataset): The testing dataset  (a MultiOmicDataset object) containing features and survival data.
        duration_col (str): Column name in the dataset for survival time.
        event_col (str): Column name in the dataset for the event occurrence (1 if event occurred, 0 otherwise).
        n_folds (int, optional): Number of folds for K-fold cross-validation. Defaults to 5.
        n_jobs (int, optional): Number of parallel jobs to run for Random Survival Forest training. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics of the RSF model, specifically the Concordance Index,
                      listed along with the method name and variable details.

    """
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
        return X, y, np.where(valid_indices)[0]

    # Prepare train and test data
    X_train, y_train, train_indices = prepare_data(train_dataset, duration_col, event_col)
    X_test, y_test, test_indices = prepare_data(test_dataset, duration_col, event_col)

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

    
    # need to get test indices to only consider samples with labels
    predicted_labels = get_predicted_labels({event_col: test_prediction}, test_dataset.subset(test_indices), 'test', 'RandomSurvivalForest')
    # Reporting
    metrics_list = [{
        'method': 'RandomSurvivalForest',
        'var': event_col, 
        'variable_type': 'numerical',
        'metric': 'cindex',
        'value': test_c_index[0]
    }]
    
    return pd.DataFrame(metrics_list), predicted_labels

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
    Plots Kaplan-Meier survival curves using plotnine and annotates log-rank test p-values.
    """
    # Prepare DataFrame
    data = pd.DataFrame({
        'Duration': durations,
        'Event': events,
        'Group': categorical_variable
    })

    kmf = KaplanMeierFitter()
    survival_curves = []

    # Fit Kaplan-Meier for each group and collect survival data
    for group in data['Group'].unique():
        group_data = data[data['Group'] == group]
        kmf.fit(group_data['Duration'], group_data['Event'], label=str(group))
        surv_df = kmf.survival_function_.reset_index()
        surv_df.columns = ['Time', 'Survival']
        surv_df['Group'] = str(group)
        survival_curves.append(surv_df)

    # Combine all curves
    plot_data = pd.concat(survival_curves)

    # Compute log-rank p-value
    categories = data['Group'].unique()
    p_text = ""
    if len(categories) == 2:
        group1 = data[data['Group'] == categories[0]]
        group2 = data[data['Group'] == categories[1]]
        result = logrank_test(group1['Duration'], group2['Duration'],
                              event_observed_A=group1['Event'],
                              event_observed_B=group2['Event'])
        p_text = f"Log-rank p = {result.p_value:.2e}"
    elif len(categories) > 2:
        result = multivariate_logrank_test(data['Duration'], data['Group'], data['Event'])
        p_text = f"Multivariate log-rank p = {result.p_value:.2e}"
    else:
        p_text = "Only one group — log-rank test not applicable"

    # Create plot
    p = (
        ggplot(plot_data, aes(x='Time', y='Survival', color='Group'))
        + geom_step()
        + labs(x='Time', y='Survival Probability', color='Group')
        + ggtitle('Kaplan-Meier Survival Curves by Group')
        + annotate("text", x=0.1, y=0.1, label=p_text, size=10, ha='left')
        + theme_minimal()
        + theme(legend_title=element_text(size=10, weight='bold'))
        + scale_color_brewer(type='qual', palette='Set1')
    )

    return p
    
def find_optimal_cutoff(expression, time, event, min_percent=0.1, max_percent=0.9, step=0.01):
    """
    Find the optimal cutoff in a continuous variable (e.g., gene expression)
    that best separates survival curves based on log-rank test.
    
    Parameters:
    - expression: pd.Series (continuous values)
    - time: survival time vector (aligned)
    - event: event indicator vector (aligned)
    - min_percent, max_percent: range of quantiles to search within
    - step: fraction of quantile steps to test

    Returns:
    - best_cutoff: value of expression that best separates survival
    - best_p: log-rank test p-value at that split
    """
    quantiles = np.arange(min_percent, max_percent, step)
    cutoffs = expression.quantile(quantiles).unique()
    
    best_p = 1
    best_cutoff = None

    for cutoff in cutoffs:
        group = expression > cutoff
        if group.nunique() < 2:
            continue
        results = logrank_test(
            time[group], time[~group],
            event[group], event[~group],
            alpha=0.99
        )
        if results.p_value < best_p:
            best_p = results.p_value
            best_cutoff = cutoff

    return best_cutoff, best_p

    
def plot_hazard_ratios(cox_model):
    """
    Plots the sorted log hazard ratios using plotnine from a fitted Cox Proportional Hazards model,
    with 95% CI and statistical significance annotations. Displays the C-index in the top-right.
    """
    # Extract summary
    coef_summary = cox_model.summary[['coef', 'coef lower 95%', 'coef upper 95%', 'p']].copy()
    coef_summary.columns = ['coef', 'coef_lower_95', 'coef_upper_95', 'p']
    coef_summary['variable'] = coef_summary.index

    # Sort by p-value
    coef_summary_sorted = coef_summary.sort_values('p').reset_index(drop=True)

    # Add significance stars
    def significance(p):
        if p < 0.0001:
            return '***'
        elif p < 0.001:
            return '**'
        elif p < 0.05:
            return '*'
        elif p < 0.1:
            return '.'
        else:
            return ''
    
    coef_summary_sorted['stars'] = coef_summary_sorted['p'].apply(significance)

    # Reverse the order for top-to-bottom importance
    coef_summary_sorted['variable'] = pd.Categorical(
        coef_summary_sorted['variable'],
        categories=coef_summary_sorted['variable'][::-1],
        ordered=True
    )
    c_index = cox_model.concordance_index_

    # Plot
    p = (
        ggplot(coef_summary_sorted, aes(x='coef', y='variable'))
        + geom_errorbarh(
            aes(xmin='coef_lower_95', xmax='coef_upper_95'),
            height=0.2, color='skyblue'
        )
        + geom_point(color='skyblue', size=3)
        + geom_text(aes(label='stars'), nudge_y=0.1, size=10)
        + annotate('vline', xintercept=0, linetype='dashed', color='gray')
        + labs(x='Log Hazard Ratio', y='', title=f'Log Hazard Ratios Sorted by P-Value with 95% CI\n Model C-index: {c_index:.2f}')
        + theme_bw()
        + theme(
            axis_text_y=element_text(size=10),
            axis_text_x=element_text(size=10),
            plot_title=element_text(weight='bold'),
        )
    )
    return p    

def build_cox_model(df, duration_col, event_col, crossval=False, n_splits=5, random_state=42):
    """
    Fits a Cox Proportional Hazards model to the data with optional cross-validation.

    Parameters:
    - df: Pandas DataFrame containing features, survival times, and event indicators.
    - duration_col: Column name for survival times.
    - event_col: Column name for event indicator (1 = event occurred, 0 = censored).
    - crossval: If True, performs K-fold cross-validation and returns average C-index.
    - n_splits: Number of folds for cross-validation.
    - random_state: Random seed for reproducibility.

    Returns:
    - cox_model: Fitted CoxPH model on the full data.
    - mean_c_index (optional): Mean C-index from cross-validation if crossval=True.
    """

    def remove_low_variance_survival_features(df, duration_col, event_col, threshold=0.01):
        events = df[event_col].astype(bool)
        low_variance_features = []

        for feature in df.drop(columns=[duration_col, event_col]).columns:
            variance_when_event = df.loc[events, feature].var()
            variance_when_no_event = df.loc[~events, feature].var()
            if variance_when_event < threshold or variance_when_no_event < threshold:
                low_variance_features.append(feature)

        df_filtered = df.drop(columns=low_variance_features)
        if low_variance_features:
            print("Removed low variance features:", low_variance_features)
        return df_filtered

    df = remove_low_variance_survival_features(df, duration_col, event_col)

    if crossval:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        c_indices = []

        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            model = CoxPHFitter()
            model.fit(train_df, duration_col=duration_col, event_col=event_col)

            # Compute risk scores on test set and C-index
            risk_scores = model.predict_partial_hazard(test_df)
            ci = concordance_index(test_df[duration_col], -risk_scores, test_df[event_col])
            c_indices.append(ci)

        mean_c_index = np.mean(c_indices)
        print(f"Cross-validated C-index (mean over {n_splits} folds): {mean_c_index:.3f}")
    else:
        mean_c_index = None

    # Fit on full data
    final_model = CoxPHFitter()
    final_model.fit(df, duration_col=duration_col, event_col=event_col)

    return final_model


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
    
def scale_and_standardize_by_labels(data_matrix, labels):
    
    """
    Scale and standardize data_matrix by factor labels.
    Data is split by factors and each subset is scaled/standardized. 

    Parameters:
    - data_matrix (numpy.ndarray): The matrix of shape (n_samples, n_features).
    - labels (numpy.ndarray): A 1D array of labels corresponding to each sample.

    Returns:
    - scaled_data_matrix (numpy.ndarray): The scaled and standardized data.
    """
    # Ensure inputs are numpy arrays
    data_matrix = np.asarray(data_matrix)
    labels = np.asarray(labels)

    # Initialize array for scaled embeddings
    scaled_data_matrix = np.zeros_like(data_matrix)

    # Process each batch independently
    unique_batches = np.unique(labels)
    for batch in unique_batches:
        # Get indices for the current batch
        batch_indices = np.where(labels == batch)[0]

        # Extract embeddings for the current batch
        batch_data = data_matrix[batch_indices, :]

        # Standardize: Zero mean, unit variance
        scaler = StandardScaler()
        batch_data_scaled = scaler.fit_transform(batch_data)

        # Assign scaled embeddings back to the result array
        scaled_data_matrix[batch_indices, :] = batch_data_scaled

    return scaled_data_matrix

# df: annotation data frame ('clin.csv')
# given a pandas data frame, go through each column and find out if the column is numeric or categorical
def get_variable_types(df):
    # Select only the categorical columns
    df_categorical = df.select_dtypes(include=['object', 'category']) 
    variable_types = {col: 'categorical' for col in df_categorical.columns}
    variable_types.update({col: 'numerical' for col in df.select_dtypes(exclude=['object', 'category']).columns})
    return variable_types

def create_covariate_matrix(covariates, variable_types, ann):
    """
    Convert clinical variables used as covariates into a covariate matrix as a Pandas DataFrame.
    Missing values in numerical variables are imputed using the median.

    Args:
        covariates (list of str): List of variable names that must exist in the "clin.csv".
        variable_types (dict): Dictionary mapping variable names to their types ('categorical' or 'numerical').
        ann (pd.DataFrame): Annotation DataFrame containing batch variable values.

    Returns:
        pd.DataFrame: A covariate matrix DataFrame where categorical variables are one-hot-encoded as 0/1 and numerical variables are imputed,
                      with features as rows and samples as columns.
    """
    covariate_features = []
    feature_names = []

    for var in covariates:
        if variable_types.get(var) == 'categorical':
            # One-hot-encode categorical variables with 0/1 encoding
            one_hot = pd.get_dummies(ann[var], prefix=var).astype(int)
            covariate_features.append(one_hot.T)  # Transpose to make features rows
            feature_names.extend(one_hot.columns.tolist())
        elif variable_types.get(var) == 'numerical':
            # Handle numerical variables with missing values
            numerical_data = ann[[var]].copy()
            # Impute missing values using the median and assign back
            numerical_data[var] = numerical_data[var].fillna(numerical_data[var].median())
            covariate_features.append(numerical_data.T)  # Transpose to make features rows
            feature_names.append(var)
        else:
            raise ValueError(f"Unknown variable type for {var}: {variable_types.get(var)}")

    # Concatenate all covariate features into a single DataFrame
    covariate_matrix = pd.concat(covariate_features, axis=0)

    # Ensure row order matches appended feature names and preserve sample names as columns
    covariate_matrix.index = feature_names
    covariate_matrix.columns = ann.index

    return covariate_matrix

def generate_synthetic_batches (n_samples_per_batch = 150,  n_features = 50):    
    # Generate batch 1 data (mean centered at 0, standard deviation 1)
    batch1_data = np.random.normal(loc=0.0, scale=1.0, size=(n_samples_per_batch, n_features))

    # Generate batch 2 data (mean shifted by +2, standard deviation 1.5)
    batch2_data = np.random.normal(loc=2.0, scale=1.5, size=(n_samples_per_batch, n_features))

    # Combine into a single dataset
    combined_data = np.vstack([batch1_data, batch2_data])
    batch_labels = np.array([0] * n_samples_per_batch + [1] * n_samples_per_batch)  # Batch labels

    # Convert to Pandas DataFrame
    feature_columns = [f"feature_{i+1}" for i in range(n_features)]
    synthetic_data = pd.DataFrame(combined_data, columns=feature_columns)
    return synthetic_data, batch_labels

def optimal_transport_align(embeddings, batch_labels, standardize_by_labels = False):
    """
    Align embeddings from two batches using Optimal Transport, preserving the order of samples.

    Parameters:
    - embeddings (pd.DataFrame): A DataFrame where rows are samples and columns are features.
    - batch_labels (np.ndarray or pd.Series): Batch labels corresponding to the rows of embeddings.

    Returns:
    - aligned_embeddings (pd.DataFrame): A DataFrame containing the aligned embeddings for all samples, with original indices preserved.
    - aligned_batch_labels (pd.Series): A Series containing the corresponding batch labels for the aligned embeddings.
    """
    # Ensure batch labels are a NumPy array
    batch_labels_np = np.array(batch_labels)

    # Identify unique batches
    unique_batches = np.unique(batch_labels_np)
    if len(unique_batches) != 2:
        raise ValueError("Optimal transport supports aligning exactly two batches.")

    # Split embeddings by batch, preserving the original indices
    batch1_indices = np.where(batch_labels_np == unique_batches[0])[0]
    batch2_indices = np.where(batch_labels_np == unique_batches[1])[0]

    batch1_embeddings = embeddings.iloc[batch1_indices].to_numpy()
    batch2_embeddings = embeddings.iloc[batch2_indices].to_numpy()

    # Compute the cost matrix (e.g., Euclidean distances)
    cost_matrix = ot.dist(batch1_embeddings, batch2_embeddings, metric='euclidean')

    # Solve the optimal transport problem
    n_samples_1 = batch1_embeddings.shape[0]
    n_samples_2 = batch2_embeddings.shape[0]
    uniform_dist_1 = np.ones(n_samples_1) / n_samples_1
    uniform_dist_2 = np.ones(n_samples_2) / n_samples_2
    transport_plan = ot.emd(uniform_dist_1, uniform_dist_2, cost_matrix)

    # Align batch 2 embeddings by transporting them to batch 1's distribution
    aligned_batch2 = np.dot(transport_plan.T, batch1_embeddings)

    # Create an array to store the aligned embeddings in the original order
    aligned_embeddings = np.zeros_like(embeddings.to_numpy())
    aligned_embeddings[batch1_indices] = batch1_embeddings
    aligned_embeddings[batch2_indices] = aligned_batch2

    # Standardize the aligned embeddings separately for each batch
    if standardize_by_labels: 
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        aligned_embeddings[batch1_indices] = scaler1.fit_transform(aligned_embeddings[batch1_indices])
        aligned_embeddings[batch2_indices] = scaler2.fit_transform(aligned_embeddings[batch2_indices])

    # Convert back to pandas DataFrame and Series, preserving indices
    aligned_embeddings_df = pd.DataFrame(aligned_embeddings, columns=embeddings.columns, index=embeddings.index)
    aligned_batch_labels = pd.Series(batch_labels, index=embeddings.index, name="batch_labels")

    return aligned_embeddings_df, aligned_batch_labels


from sklearn.neighbors import NearestNeighbors

def reciprocal_pca_mnn(embeddings, batch_labels, n_components=10, n_neighbors=5, standardize_by_labels=False, random_state=None):
    """
    Align embeddings from two batches using Reciprocal PCA (rPCA) and Mutual Nearest Neighbors (MNN).

    Parameters:
    - embeddings (pd.DataFrame): A DataFrame where rows are samples and columns are features.
    - batch_labels (np.ndarray or pd.Series): Batch labels corresponding to the rows of embeddings.
    - n_components (int): Number of principal components to use for alignment.
    - n_neighbors (int): Number of mutual nearest neighbors to use for finding anchors.
    - standardize_by_labels (bool): Whether to standardize embeddings for each batch separately.
    - random_state (int, optional): Random seed for reproducibility.

    Returns:
    - aligned_embeddings (pd.DataFrame): A DataFrame containing the aligned embeddings for all samples, with original indices preserved.
    - aligned_batch_labels (pd.Series): A Series containing the corresponding batch labels for the aligned embeddings.
    """
    # Ensure batch labels are a NumPy array
    batch_labels_np = np.array(batch_labels)

    # Identify unique batches
    unique_batches = np.unique(batch_labels_np)
    if len(unique_batches) != 2:
        raise ValueError("Reciprocal PCA supports aligning exactly two batches.")

    # Split embeddings by batch, preserving the original indices
    batch1_indices = np.where(batch_labels_np == unique_batches[0])[0]
    batch2_indices = np.where(batch_labels_np == unique_batches[1])[0]

    batch1_embeddings = embeddings.iloc[batch1_indices].to_numpy()
    batch2_embeddings = embeddings.iloc[batch2_indices].to_numpy()

    # Standardize embeddings separately for each batch if required
    if standardize_by_labels:
        batch1_embeddings = (batch1_embeddings - batch1_embeddings.mean(axis=0)) / batch1_embeddings.std(axis=0)
        batch2_embeddings = (batch2_embeddings - batch2_embeddings.mean(axis=0)) / batch2_embeddings.std(axis=0)

    # Perform PCA on both batches
    pca1 = PCA(n_components=n_components, random_state=random_state)
    pca2 = PCA(n_components=n_components, random_state=random_state)

    batch1_pca = pca1.fit_transform(batch1_embeddings)
    batch2_pca = pca2.fit_transform(batch2_embeddings)

    # Reciprocal PCA: Project each dataset into the other's PCA space
    batch1_to_batch2 = pca2.transform(batch1_embeddings)
    batch2_to_batch1 = pca1.transform(batch2_embeddings)

    # Use MNN to identify anchors
    neighbors1 = NearestNeighbors(n_neighbors=n_neighbors).fit(batch2_to_batch1)
    neighbors2 = NearestNeighbors(n_neighbors=n_neighbors).fit(batch1_to_batch2)

    distances1, indices1 = neighbors1.kneighbors(batch1_pca)
    distances2, indices2 = neighbors2.kneighbors(batch2_pca)

    # Identify mutual nearest neighbors
    mutual_anchors = []
    for i, neighbors in enumerate(indices1):
        for neighbor in neighbors:
            if i in indices2[neighbor]:
                mutual_anchors.append((i, neighbor))

    if not mutual_anchors:
        raise ValueError("No mutual nearest neighbors (MNN) found between the batches.")

    # Align the datasets using anchors
    mutual_anchors = np.array(mutual_anchors)
    batch1_anchor_indices = mutual_anchors[:, 0]
    batch2_anchor_indices = mutual_anchors[:, 1]

    # Compute the transformation to align the datasets
    batch1_aligned = batch1_pca[batch1_anchor_indices]
    batch2_aligned = batch2_pca[batch2_anchor_indices]

    alignment_matrix = np.linalg.pinv(batch2_aligned) @ batch1_aligned
    aligned_batch2 = batch2_pca @ alignment_matrix

    # Combine aligned embeddings
    aligned_embeddings = np.zeros((embeddings.shape[0], n_components))
    aligned_embeddings[batch1_indices] = batch1_pca
    aligned_embeddings[batch2_indices] = aligned_batch2

    # Convert back to pandas DataFrame and Series, preserving indices
    aligned_embeddings_df = pd.DataFrame(aligned_embeddings, 
                                         columns=[f"rPCA_{i+1}" for i in range(n_components)], 
                                         index=embeddings.index)
    aligned_batch_labels = pd.Series(batch_labels, index=embeddings.index, name="batch_labels")

    return aligned_embeddings_df, aligned_batch_labels


import tarfile
import requests
from io import StringIO

class CBioPortalData:
    def __init__(self, study_id, base_url="https://cbioportal-datahub.s3.amazonaws.com"):
        self.base_url = base_url
        self.study_id = study_id
        self.data_files = None
        self.data = None
    
    def download_study_archive(self):
        url = f"{self.base_url}/{self.study_id}.tar.gz"
        dest_file = f"{self.study_id}.tar.gz"
        
        if not os.path.exists(dest_file):
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            with open(dest_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        return dest_file
    
    def extract_archive(self, archive_path):
        base = archive_path.split(".")[0]
        
        if not os.path.exists(base):
            print(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall()
        
        self.data_files = [f for f in os.listdir(base) if f.startswith("data_") and f.endswith(".txt")]
        return base
    
    def read_data(self, files=None):
        if files is None:
            files = self.data_files
        
        data = {}
        for datatype, file in files.items():
            print(f"Importing {file}...")
            file_path = os.path.join(self.study_id, file)
            df = pd.read_csv(file_path, sep='\t', comment='#', low_memory=False)
            
            if 'mutations' in file:
                print(f"Binarizing and converting {file} to matrix...")
                df = self.binarize_mutations(df)
            elif 'clinical' not in file and 'drug_treatment' not in file:
                print(f"Converting {file} to matrix...")
                df = self.process_data(df)
            
            data[datatype] = df        
        return data
    
    def process_data(self, df):
        if 'Hugo_Symbol' in df.columns and 'Entrez_Gene_Id' in df.columns:
            df = df.drop(columns=['Entrez_Gene_Id'], errors='ignore')
        
        if 'Hugo_Symbol' in df.columns:
            df = df.drop_duplicates(subset=['Hugo_Symbol'])
            df.set_index('Hugo_Symbol', inplace=True)
        
        return df
    
    def binarize_mutations(self, df):
        required_cols = ["Hugo_Symbol", "Tumor_Sample_Barcode"]
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Can't map mutations to sample IDs. Column {col} not found.")
        
        mutation_counts = df.groupby(["Hugo_Symbol", "Tumor_Sample_Barcode"]).size().reset_index(name='count')
        mutation_matrix = mutation_counts.pivot(index='Hugo_Symbol', columns='Tumor_Sample_Barcode', values='count').fillna(0)
        mutation_matrix[mutation_matrix > 0] = 1
        
        return mutation_matrix
    
    def print_data_files(self):
        df = pd.DataFrame(self.data_files, columns=["Available Data Files"])
        print(df.to_string(index=False))
        
    def get_cbioportal_data(self, study_id, files=None):
        archive_path = self.download_study_archive()
        study_dir = self.extract_archive(archive_path)

        if files is None:
            self.print_data_files()
            print("\n\nPlease select a list of files to import. Example:\n get_cbioportal_data('study_id', files={'mut': 'data_mutations.txt', 'clin': 'data_clinical_patient.txt'})")
            return

        data = self.read_data(files)

        if 'clin' in files:
            clin = data['clin']
            clin = clin.drop_duplicates(subset=clin.columns[0])
            clin.set_index(clin.columns[0], inplace=True)
            data['clin'] = clin

        print({x: data[x].shape for x in data.keys()})
        self.data = data 
        
    def split_data(self, samples=None, ratio=0.7):
        if samples is None:
            samples = self.data['clin'].index.tolist()

        train_samples = list(pd.Series(samples).sample(frac=ratio, random_state=42))
        test_samples = list(set(samples) - set(train_samples))

        train_data = {}
        test_data = {}

        for key, df in self.data.items():
            train_data[key] = df.loc[df.index.intersection(train_samples)] if key == 'clin' else df.loc[:, df.columns.intersection(train_samples)]
            test_data[key] = df.loc[df.index.intersection(test_samples)] if key == 'clin' else df.loc[:, df.columns.intersection(test_samples)]

        return {'train': train_data, 'test': test_data}

    def print_dataset(self, dataset, outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for split, data in dataset.items():
            split_dir = os.path.join(outdir, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

            for file, df in data.items():
                df.to_csv(os.path.join(split_dir, f"{file}.csv"), sep=',')
                
    
    
def compute_correlation_loss(embeddings, batch_labels):
    # Ensure batch_labels is a float tensor
    batch_labels = batch_labels.float()

    # Normalize embeddings
    embeddings = (embeddings - embeddings.mean(dim=0, keepdim=True)) / (embeddings.std(dim=0, keepdim=True) + 1e-8)

    # Normalize batch labels
    batch_labels = (batch_labels - batch_labels.mean()) / (batch_labels.std() + 1e-8)

    # Reshape batch_labels to (num_samples, 1) for broadcasting
    batch_labels = batch_labels.unsqueeze(1)

    # Compute covariance (dot product of batch_labels and embeddings)
    covariance = torch.matmul(batch_labels.T, embeddings) / (embeddings.shape[0] - 1)

    # Compute sum of squared correlations
    loss = torch.sum(torch.abs(covariance))
    return loss

# from geomloss import SamplesLoss
def compute_transport_cost(embeddings, batch_labels, blur=0.5):
    """
    Compute a transport cost using Sinkhorn loss to align embeddings between batches.

    Parameters:
    - embeddings (torch.Tensor): Tensor of embeddings (shape: [num_samples, num_features]).
    - batch_labels (torch.Tensor): Tensor of batch labels (shape: [num_samples]).
    - blur (float): Regularization parameter for Sinkhorn OT.

    Returns:
    - loss (torch.Tensor): Sinkhorn loss value.
    """
    # Ensure batch labels are integers
    batch_labels = batch_labels.long()

    # Split embeddings by batch
    batch1_embeddings = embeddings[batch_labels == 0]
    batch2_embeddings = embeddings[batch_labels == 1]

    if batch1_embeddings.size(0) == 0 or batch2_embeddings.size(0) == 0:
        raise ValueError("Both batches must have at least one sample for transport cost computation.")

    # Initialize the Sinkhorn loss function
    loss_fn = SamplesLoss("sinkhorn", blur=blur)

    # Compute the Sinkhorn loss between the two batches
    loss = loss_fn(batch1_embeddings, batch2_embeddings)

    return loss

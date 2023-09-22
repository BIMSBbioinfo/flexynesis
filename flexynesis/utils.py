import pandas as pd
import numpy as np
import torch

from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


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

    labels = [-1 if pd.isnull(x) or x in {'nan', 'None'} else x for x in labels]

    # Add the labels to the DataFrame
    transformed_df["Label"] = labels

    if color_type == 'categorical':
        unique_labels = sorted(set(labels))
        colormap = matplotlib.cm.get_cmap("tab20", len(unique_labels))

        for i, label in enumerate(unique_labels):
            plt.scatter(
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}1"],
                transformed_df[transformed_df["Label"] == label][f"{method.upper()}2"],
                color=colormap(i),
                label=label,
                **scatter_kwargs
            )

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
    
def evaluate_classifier(y_true, y_pred):
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # F1 score (macro)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"F1 Score (Macro): {f1:.4f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Full classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    return {"balanced_acc": balanced_acc, "f1_score": f1, "kappa": kappa}

def evaluate_regressor(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "r2": r2, "pearson_corr": pearson_corr[0]}


def evaluate_wrapper(y_pred_dict, dataset):
    metrics_list = []
    for var in y_pred_dict.keys():
        ind = ~torch.isnan(dataset.ann[var])
        if dataset.variable_types[var] == 'numerical':
            metrics = evaluate_regressor(dataset.ann[var][ind], y_pred_dict[var][ind])
        else:
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

# Accepts as input a MultiomicDataset object and prints summary stats per variable 
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
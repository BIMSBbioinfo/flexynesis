import pandas as pd
import numpy as np
import torch
import math

from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

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
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "r2": r2, "pearson_corr": pearson_corr}

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

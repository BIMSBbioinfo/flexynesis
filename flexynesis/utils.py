import pandas as pd
import numpy as np
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

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

    # Add the labels to the DataFrame
    transformed_df["Label"] = labels

    labels = ['missing' if pd.isnull(x) or x in {'nan', 'None'} else x for x in labels]

    if color_type == 'categorical':
        unique_labels = list(set(labels))
        colormap = plt.cm.get_cmap("tab10", len(unique_labels))

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

    plt.legend(**legend_kwargs)
    plt.show()

def plot_true_vs_predicted(true_values, predicted_values):
    """
    Plots a scatterplot of true vs predicted values, with a regression line and annotated with the Pearson correlation coefficient.

    Args:
        true_values (list or np.array): True values
        predicted_values (list or np.array): Predicted values
    """
    # Calculate correlation coefficient
    corr, _ = pearsonr(true_values, predicted_values)
    corr_text = f"Pearson r: {corr:.2f}"
    
    # Generate scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.5)
    
    # Add regression line
    m, b = np.polyfit(true_values, predicted_values, 1)
    plt.plot(true_values, m*np.array(true_values) + b, color='red')
    
    # Add correlation text
    plt.text(min(true_values), max(predicted_values), corr_text, fontsize=12, ha='left', va='top')
    
    # Add labels and title
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    plt.show()
    
def evaluate_classifier(y_true, y_pred):
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # F1 score (macro)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score (Macro): {f1:.4f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Full classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    print(report)
    return {"balanced_acc": balanced_acc, "f1_score": f1, "kappa": kappa}

def evaluate_regressor(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "r2": r2, "pearson_corr": pearson_corr}


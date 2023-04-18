import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score


def plot_umap_scatter(df, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    """
    Plots a UMAP scatter plot from a Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame, shape (n_samples, n_features)
        The input data as a Pandas DataFrame.
    n_neighbors : int, optional, default: 15
        The number of neighbors to consider for each data point in the UMAP algorithm.
    min_dist : float, optional, default: 0.1
        The minimum distance between data points in the UMAP embedding.
    n_components : int, optional, default: 2
        The number of dimensions for the UMAP embedding (typically 2 or 3).
    metric : str, optional, default: 'euclidean'
        The distance metric to use for the UMAP algorithm.
    """

    # Convert DataFrame to NumPy array
    data = df.to_numpy()

    # Compute UMAP embedding
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = reducer.fit_transform(data)

    # Plot the UMAP scatter plot
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Scatter Plot')
    plt.show()

def plot_pca(matrix, categories):
    """
    Plots the first two principal components of the input matrix in a 2D scatter plot,
    with points colored based on the provided categories.
    
    Args:
    matrix (np.array): Input data matrix (n_samples, n_features)
    categories (list): List of categorical values (strings or integers)
    """
    # Compute PCA
    pca = PCA(n_components=2)
    transformed_matrix = pca.fit_transform(matrix)

    # Create a pandas DataFrame for easier plotting
    transformed_df = pd.DataFrame(transformed_matrix, columns=["PC1", "PC2"])

    # Add the categories to the DataFrame
    transformed_df["Category"] = categories

    # Create a unique colormap for the categories
    unique_categories = list(set(categories))
    colormap = plt.cm.get_cmap("viridis", len(unique_categories))

    # Plot the first two principal components with points colored by category
    for i, category in enumerate(unique_categories):
        plt.scatter(
            transformed_df[transformed_df["Category"] == category]["PC1"],
            transformed_df[transformed_df["Category"] == category]["PC2"],
            color=colormap(i),
            label=category,
            alpha=0.7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatter Plot with Colored Categories")
    plt.legend(title="Categories")
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
    return {"mse": mse, "r2": r2}
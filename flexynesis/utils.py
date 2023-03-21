import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt


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
    
"""
# Flexynesis
Flexynesis is a versatile Python package for various machine learning tasks, with a focus on deep learning-based model architectures, feature selection, and utility functions. Flexynesis provides a collection of pre-built models, data management tools, and utility functions that enable users to create, train, and evaluate machine learning models with ease.

# Package Contents
models_shared: Common components for model architecture and training

- data: Pytorch Dataset classes and functions to import, process multiomics data. 
- main: High-level functions for training, evaluating, and using models
- model_DirectPred: Direct prediction model architecture
- model_SVAE: Supervised Variational Autoencoder model architecture
- model_TripletEncoder: Triplet Encoder model architecture
- feature_selection: Feature selection and dimensionality reduction methods
- utils: General utility functions for data manipulation and visualization

# Main Features
- Various multi-modal data fusion methods using different kinds of deep learning architectures.
- Data management tools for loading, preprocessing, and augmenting data.
- Feature selection methods for effective dimensionality reduction.
- Utility functions to facilitate data manipulation and visualization.
- High-level functions for training, evaluating, and using models.


# Installation

To install the project using setuptools, you can follow these steps:

    1. Clone the project from the Git repository:
    ```
    git clone git@github.com:BIMSBbioinfo/flexynesis.git
    ```
    2. Navigate to the project directory:
    ```
    cd flexynesis
    ```
    3. Create a clone of the development environment, use the `spec-file.txt`:
    ```
    conda create --name flexynesis --file spec-file.txt
    conda activate flexynesis
    ```
    4. Install the project using setuptools:
    ```
    python setup.py install
    ```

# License
This package is currently private and is not meant to be used outside of Arcas.ai

# Authors
Bora Uyar, bora.uyar@mdc-berlin.de
"""
from .models_shared import *
from .data import *
from .main import *
from .model_DirectPred import *
from .model_SVAE import *
from .model_TripletEncoder import *
from .feature_selection import *
from .utils import *
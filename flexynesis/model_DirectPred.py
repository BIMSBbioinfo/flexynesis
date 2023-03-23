import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os, argparse
from scipy import stats
from functools import reduce

from .models_shared import *
    
class DirectPred(pl.LightningModule):
    """
    DirectPred is a PyTorch Lightning module for multi-omics data fusion and prediction.

    This class implements a deep learning model for fusing and predicting from multiple omics layers/matrices.
    Each omics layer is encoded separately using an MLP encoder. The resulting latent representations
    are then concatenated and passed through a fully connected network (fusion layer) to make predictions.

    Args:
        num_layers (int): Number of omics layers/matrices.
        input_dims (list of int): A list of input dimensions for each omics layer.
        latent_dim (int, optional): The dimension of the latent space for each encoder. Defaults to 16.
        num_class (int, optional): Number of output classes for the prediction task. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the MLP encoders.

    Example:

        # Instantiate a DirectPred model with 2 omics layers and input dimensions of 100 and 200
        model = DirectPred(num_layers=2, input_dims=[100, 200], latent_dim=16, num_class=1)

    """

    def __init__(self, num_layers, input_dims, latent_dim = 16, num_class = 1, **kwargs):
        super(DirectPred, self).__init__()
         # create a list of Encoder instances for separately encoding each omics layer
        self.encoders = nn.ModuleList([MLP(input_dim = input_dims[i], output_dim = latent_dim, **kwargs) for i in range(num_layers)])
        # fusion layer
        self.MLP = MLP(input_dim = latent_dim * num_layers, output_dim = num_class, **kwargs)
        self.latent_dim = latent_dim
        
    def forward(self, x_list):
        """
        Forward pass of the DirectPred model.

        Args:
            x_list (list of torch.Tensor): A list of input matrices (omics layers), one for each layer.

        Returns:
            tuple: A tuple containing the predicted output (y_pred) and a list of latent embeddings for each omics layer.
        """
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x))
        # Push concatenated encodings through a fully connected network to predict
        y_pred = self.MLP(torch.cat(embeddings_list, dim=1))
        return y_pred, embeddings_list
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the DirectPred model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Perform a single training step.

        Args:
            train_batch (tuple): A tuple containing the input data and labels for the current batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss for the current training step.
        """
        dat, y = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"train_loss": loss, "train_corr": r_value})
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            val_batch (tuple): A tuple containing the input data and labels for the current batch.
            batch_idx (int): The index of the current batch.
        """
        dat, y = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"val_loss": loss, "val_corr": r_value})
    
    def evaluate(self, dataset):
        """
        Evaluate the DirectPred model on a given dataset.

        Args:
            dataset: The dataset to evaluate the model on.

        Returns:
            float: The Pearson correlation coefficient (r_value) between the true labels and the predicted labels.
        """
        self.eval()
        layers = dataset.dat.keys()
        x_list = [dataset.dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        r_value = stats.linregress(dataset.y.detach().numpy(),
                                   torch.flatten(y_hat).detach().numpy())[2]
        return r_value




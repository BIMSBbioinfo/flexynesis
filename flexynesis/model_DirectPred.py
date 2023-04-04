import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader, random_split

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

    def __init__(self, config, dataset, task, val_size = 0.2):
        super(DirectPred, self).__init__()
        self.config = config
        self.dataset = dataset
        self.task = task
        self.val_size = val_size
        self.dat_train, self.dat_val = self.prepare_data()
        layers = list(dataset.dat.keys())
        input_dims = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        
        self.encoders = nn.ModuleList([
            MLP(input_dim=input_dims[i],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['latent_dim']) for i in range(len(layers))])
        if self.task == 'regression':
            num_class = 1
        elif self.task == 'classification':
            num_class = len(np.unique(self.dataset.y))

        self.MLP = MLP(input_dim=self.config['latent_dim'] * len(layers),
                       hidden_dim=self.config['hidden_dim'],
                       output_dim=num_class)
        
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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
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
        
        if self.task == 'regression':
            loss = F.mse_loss(torch.flatten(y_hat), y)
        elif self.task == 'classification':
            loss = F.cross_entropy(y_hat, y.long())
        self.log("train_loss", loss)
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
        if self.task == 'regression':
            loss = F.mse_loss(torch.flatten(y_hat), y)
        elif self.task == 'classification':
            loss = F.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)

    def prepare_data(self):
        lt = int(len(self.dataset)*(1-self.val_size))
        lv = len(self.dataset)-lt
        dat_train, dat_val = random_split(self.dataset, [lt, lv], 
                                          generator=torch.Generator().manual_seed(42))
        return dat_train, dat_val
    
    def train_dataloader(self):
        return DataLoader(self.dat_train, batch_size=int(self.config['batch_size']), num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dat_val, batch_size=int(self.config['batch_size']), num_workers=0, pin_memory=True, shuffle=False)
    
    def predict(self, dataset):
        """
        Evaluate the DirectPred model on a given dataset.

        Args:
            dataset: The dataset to evaluate the model on.

        Returns:
            predicted labels
        """
        self.eval()
        layers = dataset.dat.keys()
        x_list = [dataset.dat[x] for x in layers]
        y_pred = self.forward(x_list)[0].detach().numpy()
        if self.task == 'classification':
            return np.argmax(y_pred, axis=1)
        return y_pred




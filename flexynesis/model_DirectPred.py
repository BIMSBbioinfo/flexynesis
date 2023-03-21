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
    
# num_layers: number of omics layers/matrices 
class DirectPred(pl.LightningModule):
    def __init__(self, num_layers, input_dims, latent_dim = 16, num_class = 1, **kwargs):
        super(DirectPred, self).__init__()
         # create a list of Encoder instances for separately encoding each omics layer
        self.encoders = nn.ModuleList([MLP(input_dims[i], latent_dim, **kwargs) for i in range(num_layers)])
        # fusion layer
        self.MLP = MLP(num_feature = latent_dim * num_layers, num_class = num_class, h = 64)
        self.latent_dim = latent_dim
        
    def forward(self, x_list):
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x))
        # Push concatenated encodings through a fully connected network to predict
        y_pred = self.MLP(torch.cat(embeddings_list, dim=1))
        return y_pred, embeddings_list
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        dat, y = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"train_loss": loss, "train_corr": r_value})
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        dat, y = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"val_loss": loss, "val_corr": r_value})
    
    def evaluate(self, dataset):
        self.eval()
        layers = dataset.dat.keys()
        x_list = [dataset.dat[x] for x in layers]
        y_hat = self.forward(x_list)[0]
        r_value = stats.linregress(dataset.y.detach().numpy(),
                                   torch.flatten(y_hat).detach().numpy())[2]
        return r_value




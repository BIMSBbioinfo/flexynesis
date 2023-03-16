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
    
class LIMO(pl.LightningModule):
    def __init__(self, num_feature1, num_feature2, embedding_size = 16, num_class = 1, **kwargs):
        super(LIMO, self).__init__()
        self.omics1 = MLP(num_feature1, embedding_size, **kwargs)
        self.omics2 = MLP(num_feature2, embedding_size, **kwargs)
        self.MLP = MLP(num_feature = embedding_size * 2, num_class = num_class, h = 64)
        self.embedding_size = embedding_size
        
    def forward(self, dat_omics1, dat_omics2):
        l1 = self.omics1(dat_omics1) #embeddings from layer1 (e.g. gex)
        l2 = self.omics2(dat_omics2) #embeddings from layer2 (e.g. cnv)
        #run the regressor on concatenated encodings 
        l = torch.cat((l1, l2), dim = 1)
        y_pred = self.MLP(l)
        return y_pred, l1, l2
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.004048966088420985)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        dat, y = train_batch
        omics = list(dat.keys())
        y_hat = self.forward(dat[omics[0]], 
                             dat[omics[1]])[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"train_loss": loss, "train_corr": r_value})
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        dat, y = val_batch
        omics = list(dat.keys())
        y_hat = self.forward(dat[omics[0]], 
                                     dat[omics[1]])[0]
        loss = F.mse_loss(torch.flatten(y_hat), y)
        r_value = stats.linregress(y.detach().numpy(), torch.flatten(y_hat).detach().numpy())[2]
        self.log_dict({"val_loss": loss, "val_corr": r_value})
    
    def evaluate(self, dataset):
        self.eval()
        omics = list(dataset.dat.keys())
        y_hat = self.forward(dataset.dat[omics[0]], 
                             dataset.dat[omics[1]])[0]
        r_value = stats.linregress(dataset.y.detach().numpy(),
                                   torch.flatten(y_hat).detach().numpy())[2]
        return r_value




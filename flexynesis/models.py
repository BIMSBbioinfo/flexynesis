# Late integration for multi-omics (limo): like moli without triplet loss

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


class MultiomicDataset(Dataset):
    def __init__(self, dat, y, features, samples):
        self.dat = dat #dict with multiple matrices
        self.y = y #shared labels for all matrices 
        self.features = features
        self.samples = samples 
    def __getitem__(self, index):
        return {x: self.dat[x][index] for x in self.dat.keys()}, self.y[index]
    def __len__ (self):
        return len(self.y)

# a MLP model for regression/classification
# set num_class to 1 for regression. num_class > 1 => classification
class MLP(nn.Module):
    def __init__(self, num_feature, num_class, h = 32):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(num_feature, h)
        self.layer_out = nn.Linear(h, num_class)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.4)
        self.batchnorm = nn.BatchNorm1d(h)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

    
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

def get_labels(dat, drugs, drugName, batch_size):
    y = drugs[drugName]
    y = y[~y.isna()]

    # list of samples in the assays
    samples = list(reduce(set.intersection, [set(item) for item in [dat[x].columns for x in dat.keys()]]))
    # keep samples with labels
    samples = list(set(y.index).intersection(samples))
    if len(samples) % batch_size == 1:
        # I do this to avoid batches of size 1
        samples = samples[0:len(samples)-1]
    #subset assays and labels for the remaining samples
    dat = {x: dat[x][samples] for x in dat.keys()}
    y = y[samples]
    return dat, y

# dat: list of matrices, features on the rows, samples on the columns
# ann: pandas data frame with 'y' as variable for the corresponding samples on the matrix columns
# task_type: type of outcome (classification/regression)
# notice the tables are transposed to return samples on rows, features on columns
def get_torch_dataset(dat, labels):
    # keep a copy of row/column names
    features = {x: dat[x].index for x in dat.keys()}
    samples = {x: dat[x].columns for x in dat.keys()}
    dat = {x: torch.from_numpy(np.array(dat[x].T)).float() for x in dat.keys()}
    y =  torch.from_numpy(np.array(labels)).float()
    return MultiomicDataset(dat, y, features, samples)

def make_dataset(dat, *args):
    dat, y = get_labels(dat, *args)
    dataset = get_torch_dataset(dat, y)
    return dataset

# warning: no validation loops
# only regression models for now
def train_model(dataset, n_epoch, embedding_size, batch_size, model = None, val_size = 0):
    if model is None:
        # model
        f1, f2 = [dataset.dat[omics].shape[1] for omics in dataset.dat.keys()]
        model = LIMO(f1, f2 , h = 16, embedding_size=embedding_size, num_class = 1)
    # training
    if val_size == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, limit_val_batches = 0, num_sanity_val_steps = 0, 
                             strategy=DDPStrategy(find_unused_parameters=False), num_nodes = 4) 
        trainer.fit(model, train_loader) 
    elif val_size > 0:
        # split train into train/val
        dat_train, dat_val = random_split(dataset, [1-val_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(dat_train, batch_size=batch_size, num_workers = 0)
        val_loader = DataLoader(dat_val, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, 
                             strategy=DDPStrategy(find_unused_parameters=False), 
                             num_nodes = 4) 
        trainer.fit(model, train_loader, val_loader) 
    return model
    
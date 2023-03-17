# Supervised VAE-MMD architecture
import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from scipy import stats

from .models_shared import *

class supervised_vae_ef(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, latent_dim, num_class, **kwargs):
        super(supervised_vae_ef, self).__init__()
        self.Encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        self.MLP = MLP(latent_dim, num_class, **kwargs)
        self.latent_dim = latent_dim
        self.num_class = num_class
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)       
        z = mean + var*epsilon                         
        return z
                
    def forward(self, X):
        mean, log_var = self.Encoder(X)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        X_hat = self.Decoder(z)
        #run the supervisor 
        y_pred = self.MLP(z)
        return X_hat, z, mean, log_var, y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        dat, y = train_batch
        X = dat['all'] # assume all omics layers are concatenated with the key 'all'
        mean, log_var = self.Encoder(X)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        X_hat = self.Decoder(z)
        y_pred = self.MLP(z)
        
        mmd_loss = self.MMD_loss(z.shape[1], z, X_hat, X)
        sp_loss = F.mse_loss(y_pred, y)
        loss = mmd_loss + sp_loss
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        dat, y = val_batch
        X = dat['all'] # assume all omics layers are concatenated with the key 'all'
        mean, log_var = self.Encoder(X)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        X_hat = self.Decoder(z)
        y_pred = self.MLP(z)
        
        mmd_loss = self.MMD_loss(z.shape[1], z, X_hat, X)
        sp_loss = F.mse_loss(y_pred, y)
        loss = mmd_loss + sp_loss
        
        self.log('val_loss', loss)
    
    def transform(self, dataset):
        self.eval()
        X = dataset.dat['all'] # assume all omics layers are concatenated with the key 'all'
        M = torch.from_numpy(np.array(X)).float()
        z = pd.DataFrame(self.forward(M)[1].detach().numpy())
        z.columns = [''.join(['LF', str(x+1)]) for x in z.columns]
        z.index = dataset.samples['all']
        return z
    
    def evaluate(self, dataset):
        self.eval()
        X = dataset.dat['all'] # assume all omics layers are concatenated with the key 'all'
        X_hat, z, mean, log_var, y_pred = self.forward(X)
        r_value = stats.linregress(dataset.y.detach().numpy(),
                                   torch.flatten(y_pred).detach().numpy())[2]
        return r_value

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def MMD_loss(self, latent_dim, z, xhat, x):
        true_samples = torch.randn(200, latent_dim)
        mmd = self.compute_mmd(true_samples, z) # compute maximum mean discrepancy (MMD)
        nll = (xhat - x).pow(2).mean() #negative log likelihood
        return mmd+nll
    
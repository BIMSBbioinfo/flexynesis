# Supervised VAE-MMD architecture
import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from scipy import stats

from .models_shared import *

# Supervised Variational Auto-encoder that can train one or more layers of omics datasets 
# num_layers: number of omics layers in the input
# each layer is encoded separately, encodings are concatenated, and decoded separately 
# depends on MLP, Encoder, Decoder classes in models_shared
class supervised_vae(pl.LightningModule):
    def __init__(self, num_layers, input_dims, hidden_dims, latent_dim, num_class, **kwargs):
        super(supervised_vae, self).__init__()
        
        # define supervisor head
        self.MLP = MLP(latent_dim, num_class, **kwargs)
        self.latent_dim = latent_dim
        self.num_class = num_class
        
        # create a list of Encoder instances for separately encoding each omics layer
        self.encoders = nn.ModuleList([Encoder(input_dims[i], hidden_dims, latent_dim) for i in range(num_layers)])
        # Fully connected layers for concatenated means and log_vars
        self.FC_mean = nn.Linear(num_layers * latent_dim, latent_dim)
        self.FC_log_var = nn.Linear(num_layers * latent_dim, latent_dim)
        
        # list of decoders to decode each omics layer separately 
        self.decoders = nn.ModuleList([Decoder(latent_dim, hidden_dims[::-1], input_dims[i]) for i in range(num_layers)])

    def multi_encoder(self, x_list):
        means, log_vars = [], []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            mean, log_var = self.encoders[i](x)
            means.append(mean)
            log_vars.append(log_var)

        # Concatenate means and log_vars
        # Push concatenated means and log_vars through the fully connected layers
        mean = self.FC_mean(torch.cat(means, dim=1))
        log_var = self.FC_log_var(torch.cat(log_vars, dim=1))
        return mean, log_var
    
    def forward(self, x_list):
        mean, log_var = self.multi_encoder(x_list)
        
        # generate latent layer
        z = self.reparameterization(mean, log_var)

        # Decode each latent variable with its corresponding Decoder
        x_hat_list = [self.decoders[i](z) for i in range(len(x_list))]

        #run the supervisor 
        y_pred = self.MLP(z)
        
        return x_hat_list, z, mean, log_var, y_pred
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)       
        z = mean + var*epsilon                         
        return z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        dat, y = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        mean, log_var = self.multi_encoder(x_list)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat_list = [self.decoders[i](z) for i in range(len(x_list))]
        y_pred = self.MLP(z)
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        sp_loss = F.mse_loss(y_pred, y)
        loss = mmd_loss + sp_loss
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        dat, y = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        mean, log_var = self.multi_encoder(x_list)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat_list = [self.decoders[i](z) for i in range(len(x_list))]
        y_pred = self.MLP(z)
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        sp_loss = F.mse_loss(y_pred, y)
        loss = mmd_loss + sp_loss
        
        self.log('val_loss', loss)
        return loss
    
    def transform(self, dataset):
        self.eval()
        layers = list(dataset.dat.keys())
        x_list = [dataset.dat[x] for x in layers]
        M = self.forward(x_list)[1].detach().numpy()
        z = pd.DataFrame(M)
        z.columns = [''.join(['LF', str(x+1)]) for x in z.columns]
        z.index = dataset.samples
        return z
    
    def evaluate(self, dataset):
        self.eval()
        layers = list(dataset.dat.keys())
        x_list = [dataset.dat[x] for x in layers]
        X_hat, z, mean, log_var, y_pred = self.forward(x_list)
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


    

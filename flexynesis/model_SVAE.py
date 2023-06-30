# Supervised VAE-MMD architecture
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

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
    """
    Supervised Variational Auto-encoder for multi-omics data fusion and prediction.

    This class implements a deep learning model for fusing and predicting from multiple omics layers/matrices.
    Each omics layer is encoded separately using an Encoder. The resulting latent representations are then
    concatenated and passed through a fully connected network (fusion layer) to make predictions. The model
    also includes a supervisor head for supervised learning.

    Args:
        num_layers (int): Number of omics layers/matrices.
        input_dims (list of int): A list of input dimensions for each omics layer.
        hidden_dims (list of int): A list of hidden dimensions for the Encoder and Decoder.
        latent_dim (int): The dimension of the latent space for each encoder.
        num_class (int): Number of output classes for the prediction task.
        **kwargs: Additional keyword arguments to be passed to the MLP encoders.

    Example:

        # Instantiate a supervised_vae model with 2 omics layers and input dimensions of 100 and 200
        model = supervised_vae(num_layers=2, input_dims=[100, 200], hidden_dims=[64, 32], latent_dim=16, num_class=1)

    """
    def __init__(self,  config, dataset, target_variables, batch_variables = None, val_size = 0.2):
        super(supervised_vae, self).__init__()
        self.config = config
        self.dataset = dataset
        self.target_variables = target_variables
        self.batch_variables = batch_variables
        self.variables = target_variables + batch_variables if batch_variables else target_variables
        self.val_size = val_size

        self.dat_train, self.dat_val = self.prepare_data()
        # sometimes the model may have exploding/vanishing gradients leading to NaN values
        self.nan_detected = False 
        
        layers = list(dataset.dat.keys())
        input_dims = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        # create a list of Encoder instances for separately encoding each omics layer
        self.encoders = nn.ModuleList([Encoder(input_dims[i], [config['hidden_dim']], config['latent_dim']) for i in range(len(layers))])
        # Fully connected layers for concatenated means and log_vars
        self.FC_mean = nn.Linear(len(layers) * config['latent_dim'], config['latent_dim'])
        self.FC_log_var = nn.Linear(len(layers) * config['latent_dim'], config['latent_dim'])
        # list of decoders to decode each omics layer separately 
        self.decoders = nn.ModuleList([Decoder(config['latent_dim'], [config['hidden_dim']], input_dims[i]) for i in range(len(layers))])

        # define supervisor heads
        # using ModuleDict to store multiple MLPs
        self.MLPs = nn.ModuleDict()         
        for var in self.variables:
            if self.dataset.variable_types[var] == 'numerical':
                num_class = 1
            else:
                num_class = len(np.unique(self.dataset.ann[var]))
            self.MLPs[var] = MLP(input_dim = config['latent_dim'], 
                                 hidden_dim = config['supervisor_hidden_dim'], 
                                 output_dim = num_class)
                                       
    def multi_encoder(self, x_list):
        """
        Encode each input matrix separately using the corresponding Encoder.

        Args:
            x_list (list of torch.Tensor): List of input matrices for each omics layer.

        Returns:
            tuple: Tuple containing:
                - mean (torch.Tensor): Concatenated mean values from each encoder.
                - log_var (torch.Tensor): Concatenated log variance values from each encoder.
        """
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
        """
        Forward pass through the model.

        Args:
            x_list (list of torch.Tensor): List of input matrices for each omics layer.

        Returns:
            tuple: Tuple containing:
                - x_hat_list (list of torch.Tensor): List of reconstructed matrices for each omics layer.
                - z (torch.Tensor): Latent representation.
                - mean (torch.Tensor): Concatenated mean values from each encoder.
                - log_var (torch.Tensor): Concatenated log variance values from each encoder.
                - y_pred (torch.Tensor): Predicted output.
        """
        mean, log_var = self.multi_encoder(x_list)
        
        # generate latent layer
        z = self.reparameterization(mean, log_var)

        # Decode each latent variable with its corresponding Decoder
        x_hat_list = [self.decoders[i](z) for i in range(len(x_list))]

        #run the supervisor heads using the latent layer as input
        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(z)
            
        return x_hat_list, z, mean, log_var, outputs
        
    def reparameterization(self, mean, var):
        """
        Reparameterize the mean and variance values.

        Args:
            mean (torch.Tensor): Mean values from the encoders.
            var (torch.Tensor): Variance values from the encoders.

        Returns:
            torch.Tensor: Latent representation.
        """
        epsilon = torch.randn_like(var)       
        z = mean + var*epsilon                         
        return z
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Adam: Adam optimizer with learning rate 1e-3.
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
            torch.Tensor: The total loss for the current training step.
        """
        dat, y_dict = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        
        x_hat_list, z, mean, log_var, outputs = self.forward(x_list)
        
        # Check for NaNs in the latent space
        if torch.isnan(z).any():
            raise ValueError("NaN value detected in latent factors")
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        # compute loss values for the supervisor heads 
        total_loss = 0        
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]

            if self.dataset.variable_types[var] == 'numerical':
                # Ignore instances with missing labels for numerical variables
                valid_indices = ~torch.isnan(y)
                if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                    y_hat = y_hat[valid_indices]
                    y = y[valid_indices]
                    
                    loss = F.mse_loss(torch.flatten(y_hat), y.float())
                    if self.batch_variables is not None and var in self.batch_variables:
                        y_shuffled = y[torch.randperm(len(y))]
                        # compute the difference between prediction error 
                        # when using actual labels and shuffled labels 
                        loss_shuffled = F.mse_loss(torch.flatten(y_hat), y_shuffled.float())
                        loss = torch.abs(loss - loss_shuffled)
                        
                    self.log(f"train_loss_{var}", loss)
                    total_loss += loss
                else:
                    total_loss += 0.0  # if no valid labels, set loss to 0
            else:
                # Ignore instances with missing labels for categorical variables
                # Assuming that missing values were encoded as -1
                valid_indices = (y != -1) & (~torch.isnan(y))
                if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                    y_hat = y_hat[valid_indices]
                    y = y[valid_indices]
                    loss = F.cross_entropy(y_hat, y.long())
                    
                    if self.batch_variables is not None and var in self.batch_variables:
                        y_shuffled = y[torch.randperm(len(y))]
                        # compute the difference between prediction error 
                        # when using actual labels and shuffled labels 
                        loss_shuffled = F.cross_entropy(y_hat, y_shuffled.long())
                        loss = torch.abs(loss - loss_shuffled)
                        
                    self.log(f"train_loss_{var}", loss)
                    total_loss += loss
                else:
                    total_loss += 0.0  # if no valid labels, set loss to 0
        
        train_loss = mmd_loss + total_loss
        
        self.log_dict({'train_loss': train_loss, 'mmd': mmd_loss, 'sp': total_loss}, prog_bar=True)
        return train_loss
    
    def validation_step(self, val_batch, batch_idx):
        """
        Perform a single training step.

        Args:
            val_batch (tuple): A tuple containing the input data and labels for the current batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The total loss for the current training step.
        """
        dat, y_dict = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        
        x_hat_list, z, mean, log_var, outputs = self.forward(x_list)
        
        # Check for NaNs in the latent space
        if torch.isnan(z).any():
            raise ValueError("NaN value detected in latent factors")
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        # compute loss values for the supervisor heads 
        total_loss = 0        
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]

            if self.dataset.variable_types[var] == 'numerical':
                # Ignore instances with missing labels for numerical variables
                valid_indices = ~torch.isnan(y)
                if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                    y_hat = y_hat[valid_indices]
                    y = y[valid_indices]
                    
                    loss = F.mse_loss(torch.flatten(y_hat), y.float())
                    if self.batch_variables is not None and var in self.batch_variables:
                        y_shuffled = y[torch.randperm(len(y))]
                        # compute the difference between prediction error 
                        # when using actual labels and shuffled labels 
                        loss_shuffled = F.mse_loss(torch.flatten(y_hat), y_shuffled.float())
                        loss = torch.abs(loss - loss_shuffled)
                        
                    self.log(f"train_loss_{var}", loss)
                    total_loss += loss
                else:
                    total_loss += 0.0  # if no valid labels, set loss to 0
            else:
                # Ignore instances with missing labels for categorical variables
                # Assuming that missing values were encoded as -1
                valid_indices = (y != -1) & (~torch.isnan(y))
                if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                    y_hat = y_hat[valid_indices]
                    y = y[valid_indices]
                    loss = F.cross_entropy(y_hat, y.long())
                    
                    if self.batch_variables is not None and var in self.batch_variables:
                        y_shuffled = y[torch.randperm(len(y))]
                        # compute the difference between prediction error 
                        # when using actual labels and shuffled labels 
                        loss_shuffled = F.cross_entropy(y_hat, y_shuffled.long())
                        loss = torch.abs(loss - loss_shuffled)
                        
                    self.log(f"train_loss_{var}", loss)
                    total_loss += loss
                else:
                    total_loss += 0.0  # if no valid labels, set loss to 0
        
        val_loss = mmd_loss + total_loss
        
        self.log_dict({'val_loss': val_loss, 'mmd': mmd_loss, 'sp': total_loss}, prog_bar=True)
        return val_loss
                                       
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
        
    def transform(self, dataset):
        """
        Transform the input dataset to latent representation.

        Args:
            dataset (MultiOmicDataset): MultiOmicDataset containing input matrices for each omics layer.

        Returns:
            pd.DataFrame: Transformed dataset as a pandas DataFrame.
        """
        self.eval()
        layers = list(dataset.dat.keys())
        x_list = [dataset.dat[x] for x in layers]
        M = self.forward(x_list)[1].detach().numpy()
        z = pd.DataFrame(M)
        z.columns = [''.join(['LF', str(x+1)]) for x in z.columns]
        z.index = dataset.samples
        return z
    
    def predict(self, dataset):
        """
        Evaluate the model on a dataset.

        Args:
            dataset (CustomDataset): Custom dataset containing input matrices for each omics layer.

        Returns:
            predicted values.
        """
        self.eval()
        layers = list(dataset.dat.keys())
        x_list = [dataset.dat[x] for x in layers]
        X_hat, z, mean, log_var, outputs = self.forward(x_list)
        
        predictions = {}
        for var in self.variables:
            y_pred = outputs[var].detach().numpy()
            if self.dataset.variable_types[var] == 'categorical':
                predictions[var] = np.argmax(y_pred, axis=1)
            else:
                predictions[var] = y_pred

        return predictions
    
    def compute_kernel(self, x, y):
        """
        Compute the Gaussian kernel matrix between two sets of vectors.

        Args:
            x (torch.Tensor): A tensor of shape (x_size, dim) representing the first set of vectors.
            y (torch.Tensor): A tensor of shape (y_size, dim) representing the second set of vectors.

        Returns:
            torch.Tensor: The Gaussian kernel matrix of shape (x_size, y_size) computed between x and y.
        """
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
        """
        Compute the maximum mean discrepancy (MMD) between two sets of vectors.

        Args:
            x (torch.Tensor): A tensor of shape (x_size, dim) representing the first set of vectors.
            y (torch.Tensor): A tensor of shape (y_size, dim) representing the second set of vectors.

        Returns:
            torch.Tensor: A scalar tensor representing the MMD between x and y.
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def MMD_loss(self, latent_dim, z, xhat, x):
        """
        Compute the loss function based on maximum mean discrepancy (MMD) and negative log likelihood (NLL).

        Args:
            latent_dim (int): The dimensionality of the latent space.
            z (torch.Tensor): A tensor of shape (batch_size, latent_dim) representing the latent codes.
            xhat (torch.Tensor): A tensor of shape (batch_size, dim) representing the reconstructed data.
            x (torch.Tensor): A tensor of shape (batch_size, dim) representing the original data.

        Returns:
            torch.Tensor: A scalar tensor representing the MMD loss.
        """
        true_samples = torch.randn(200, latent_dim)
        mmd = self.compute_mmd(true_samples, z) # compute maximum mean discrepancy (MMD)
        nll = (xhat - x).pow(2).mean() #negative log likelihood
        return mmd+nll


    

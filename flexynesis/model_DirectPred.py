import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
import os, argparse
from scipy import stats
from functools import reduce

from captum.attr import IntegratedGradients

from .models_shared import *



class DirectPred(pl.LightningModule):
    def __init__(self, config, dataset, target_variables, batch_variables = None, val_size = 0.2):
        super(DirectPred, self).__init__()
        self.config = config
        self.dataset = dataset
        self.target_variables = target_variables
        self.batch_variables = batch_variables
        self.variables = target_variables + batch_variables if batch_variables else target_variables
        self.val_size = val_size
        self.dat_train, self.dat_val = self.prepare_data()
        
        layers = list(dataset.dat.keys())
        input_dims = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        
        self.encoders = nn.ModuleList([
            MLP(input_dim=input_dims[i],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['latent_dim']) for i in range(len(layers))])

        self.MLPs = nn.ModuleDict() # using ModuleDict to store multiple MLPs
        for var in self.variables:
            if self.dataset.variable_types[var] == 'numerical':
                num_class = 1
            else:
                num_class = len(np.unique(self.dataset.ann[var]))
            self.MLPs[var] = MLP(input_dim=self.config['latent_dim'] * len(layers),
                                         hidden_dim=self.config['hidden_dim'],
                                         output_dim=num_class)

    def forward(self, x_list):
        """
        Forward pass of the DirectPred model.

        Args:
            x_list (list of torch.Tensor): A list of input matrices (omics layers), one for each layer.

        Returns:
            dict: A dictionary where each key-value pair corresponds to the target variable name and its predicted output respectively.
        """
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings_concat)
        return outputs  
    
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the DirectPred model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def compute_loss(self, var, y, y_hat):
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
            else:
                loss = 0 # if no valid labels, set loss to 0
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
            else: 
                loss = 0
        return loss

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
        outputs = self.forward(x_list)
        total_loss = 0        
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]

            loss = self.compute_loss(var, y, y_hat)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            total_loss += loss            
        return total_loss

    
    def validation_step(self, val_batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            val_batch (tuple): A tuple containing the input data and labels for the current batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The total loss for the current validation step.
        """
        dat, y_dict = val_batch       
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        outputs = self.forward(x_list)
        total_loss = 0        
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]

            loss = self.compute_loss(var, y, y_hat)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            total_loss += loss            
        return total_loss

    
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
            A dictionary where each key is a target variable and the corresponding value is the predicted output for that variable.
        """
        self.eval()
        layers = dataset.dat.keys()
        x_list = [dataset.dat[x] for x in layers]
        outputs = self.forward(x_list)

        predictions = {}
        for var in self.variables:
            y_pred = outputs[var].detach().numpy()
            if self.dataset.variable_types[var] == 'categorical':
                predictions[var] = np.argmax(y_pred, axis=1)
            else:
                predictions[var] = y_pred
        return predictions

    def transform(self, dataset):
        """
        Transform the input data into a lower-dimensional space using the trained encoders.

        Args:
            dataset: The input dataset containing the omics data.

        Returns:
            pd.DataFrame: A dataframe of embeddings where the row indices are 
                          dataset.samples and the column names are created by appending 
                          the substring "E" to each dimension index.
        """
        self.eval()
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(dataset.dat.values()):
            embeddings_list.append(self.encoders[i](x))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        # Converting tensor to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(embeddings_concat.detach().numpy(), 
                                     index=dataset.samples,
                                     columns=[f"E{dim}" for dim in range(embeddings_concat.shape[1])])
        return embeddings_df
        
    # Adaptor forward function for captum integrated gradients. 
    def forward_target(self, *args):
        input_data = list(args[:-2])  # one or more tensors (one per omics layer)
        target_var = args[-2]  # target variable of interest
        steps = args[-1]  # number of steps for IntegratedGradients().attribute 
        outputs_list = []
        for i in range(steps):
            # get list of tensors for each step into a list of tensors
            x_step = [input_data[j][i] for j in range(len(input_data))]
            out = self.forward(x_step)
            outputs_list.append(out[target_var])
        return torch.cat(outputs_list, dim = 0)
        
    def compute_feature_importance(self, target_var, steps = 5):
        """
        Compute the feature importance.

        Args:
            input_data (torch.Tensor): The input data to compute the feature importance for.
            target_var (str): The target variable to compute the feature importance for.
        Returns:
            attributions (list of torch.Tensor): The feature importances for each class.
        """
        x_list = [self.dataset.dat[x] for x in self.dataset.dat.keys()]
                
        # Initialize the Integrated Gradients method
        ig = IntegratedGradients(self.forward_target)

        input_data = tuple([data.unsqueeze(0).requires_grad_() for data in x_list])

        # Define a baseline (you might need to adjust this depending on your actual data)
        baseline = tuple([torch.zeros_like(data) for data in input_data])

        # Get the number of classes for the target variable
        if self.dataset.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique(self.dataset.ann[target_var]))

        # Compute the feature importance for each class
        attributions = []
        if num_class > 1:
            for target_class in range(num_class):
                attributions.append(ig.attribute(input_data, baseline, additional_forward_args=(target_var, steps), target=target_class, n_steps=steps))
        else:
            attributions.append(ig.attribute(input_data, baseline, additional_forward_args=(target_var, steps), n_steps=steps))

        # summarize feature importances
        # Compute absolute attributions
        abs_attr = [[torch.abs(a) for a in attr_class] for attr_class in attributions]
        # average over samples 
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]

        # combine into a single data frame 
        df_list = []
        layers = list(self.dataset.dat.keys())
        for i in range(num_class):
            for j in range(len(layers)):
                features = self.dataset.features[layers[j]]
                importances = imp[i][j][0].detach().numpy()
                df_list.append(pd.DataFrame({'target_variable': target_var, 'target_class': i, 'layer': layers[j], 'name': features, 'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index = True)
        return df_imp



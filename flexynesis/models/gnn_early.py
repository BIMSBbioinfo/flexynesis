import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split

import lightning as pl

from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients

from ..modules import MLP, cox_ph_loss, GNNs


class GNNEarly(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset, # MultiOmicGeometricDataset object
        target_variables,
        batch_variables=None,
        surv_event_var=None,
        surv_time_var=None,
        use_loss_weighting=True,
        device_type = None,
        gnn_conv_type = None 
    ):
        super().__init__()
        self.config = config
        self.target_variables = target_variables
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        # both surv event and time variables are assumed to be numerical variables
        # we create only one survival variable for the pair (surv_time_var and surv_event_var)
        if self.surv_event_var is not None and self.surv_time_var is not None:
            self.target_variables = self.target_variables + [self.surv_event_var]
        self.batch_variables = batch_variables
        self.variables = self.target_variables + self.batch_variables if self.batch_variables else self.target_variables
        self.variable_types = dataset.multiomic_dataset.variable_types 
        self.ann = dataset.multiomic_dataset.ann 
        self.edge_index = dataset.edge_index
        
        self.feature_importances = {}
        self.use_loss_weighting = use_loss_weighting

        self.device_type = device_type 
        self.gnn_conv_type = gnn_conv_type
        
        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.edge_index = self.edge_index.to(device) # edge index is re-used across samples, so we keep it in device
                
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for var in self.variables:
                self.log_vars[var] = nn.Parameter(torch.zeros(1))
        
        node_features = dataset[0][0].shape[1] # number of node features
        node_count = dataset[0][0].shape[0] #number of nodes
        self.encoders = GNNs(
                        input_dim=node_features,
                        hidden_dim=int(self.config["hidden_dim_factor"] * node_count),  
                        output_dim=self.config["latent_dim"],
                        act = self.config['activation'],
                        conv = self.gnn_conv_type
        )

        # Init output layers
        self.MLPs = nn.ModuleDict()
        for var in self.variables:
            if self.variable_types[var] == "numerical":
                num_class = 1
            else:
                num_class = len(np.unique(self.ann[var]))
            self.MLPs[var] = MLP(
                input_dim=self.config["latent_dim"],
                hidden_dim=self.config["supervisor_hidden_dim"],
                output_dim=num_class
            )
            
    def forward(self, x, edge_index): 
        embeddings = self.encoders(x, edge_index)
        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings)
        return outputs

            
    def training_step(self, batch):
        x, y_dict, samples = batch
        outputs = self.forward(x, self.edge_index)

        losses = {}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var]
                risk_scores = outputs[var]
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = self.compute_total_loss(losses)
        losses["train_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return total_loss

    def validation_step(self, batch):
        x, y_dict, samples = batch
        outputs = self.forward(x, self.edge_index)
        losses = {}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var]
                risk_scores = outputs[var]
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = sum(losses.values())
        losses["val_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return total_loss
            
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config["lr"])
        return optimizer

    def compute_loss(self, var, y, y_hat):
        if self.variable_types[var] == "numerical":
            # Ignore instances with missing labels for numerical variables
            valid_indices = ~torch.isnan(y)
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.mse_loss(torch.flatten(y_hat), y.float())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)  # if no valid labels, set loss to 0
        else:
            # Ignore instances with missing labels for categorical variables
            # Assuming that missing values were encoded as -1
            valid_indices = (y != -1) & (~torch.isnan(y))
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.cross_entropy(y_hat, y.long())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        return loss

    def compute_total_loss(self, losses):
        if self.use_loss_weighting and len(losses) > 1:
            # Compute weighted loss for each loss
            # Weighted loss = precision * loss + log-variance
            total_loss = sum(
                torch.exp(-self.log_vars[name]) * loss + self.log_vars[name] for name, loss in losses.items()
            )
        else:
            # Compute unweighted total loss
            total_loss = sum(losses.values())
        return total_loss
    
    def predict(self, dataset):
        self.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the appropriate device

        # Create a DataLoader with a practical batch size
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust the batch size as needed
        edge_index = dataset.edge_index.to(device)  # Move edge_index to GPU

        predictions = {var: [] for var in self.variables}  # Initialize prediction storage

        # Process each batch
        for x, y_dict,samples in dataloader:
            x = x.to(device)  # Move data to GPU

            outputs = self.forward(x, edge_index)

            # Collect predictions for each variable
            for var in self.variables:
                y_pred = outputs[var].detach().cpu().numpy()  # Move outputs back to CPU and convert to numpy
                if self.variable_types[var] == "categorical":
                    predictions[var].extend(np.argmax(y_pred, axis=1))
                else:
                    predictions[var].extend(y_pred)

        # Convert lists to arrays if necessary, depending on the downstream use-case
        predictions = {var: np.array(predictions[var]) for var in predictions}

        return predictions

    def transform(self, dataset):
        self.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the appropriate device

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust the batch size as needed
        all_embeddings = []  # List to store embeddings from all batches
        sample_ids = []  # List to store indices for all samples processed

        # Process each batch
        for x, y_dict, samples in dataloader:
            x = x.to(device)  # Move data to GPU
            edge_index = dataset.edge_index.to(device)  # Move edge_index to GPU

            embeddings = self.encoders(x, edge_index).detach().cpu().numpy()  # Compute embeddings and move to CPU
            all_embeddings.append(embeddings)
            sample_ids.extend(samples)  
            
        # Concatenate all embeddings into a single numpy array
        all_embeddings = np.vstack(all_embeddings)

        # Converting tensor to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(
            all_embeddings,
            index=sample_ids,  # Use the correct indices as row names
            columns=[f"E{dim}" for dim in range(all_embeddings.shape[1])],
        )
        return embeddings_df

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
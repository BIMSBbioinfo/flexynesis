import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl

import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
from captum.attr import IntegratedGradients

from ..modules import GCNN, MLP


class DirectPredGCNN(pl.LightningModule):
    def __init__(self, config, dataset, target_variables, batch_variables=None, val_size=0.2, use_loss_weighting=True):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.target_variables = target_variables
        self.batch_variables = batch_variables
        self.variables = target_variables + batch_variables if batch_variables else target_variables
        self.val_size = val_size
        self.dat_train, self.dat_val = self.prepare_data()
        self.feature_importances = {}
        self.use_loss_weighting = use_loss_weighting

        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for var in self.variables:
                self.log_vars[var] = nn.Parameter(torch.zeros(1))

        # Init modality encoders
        layers = list(self.dataset.dat.keys())
        # NOTE: For now we use matrices, so number of node input features is 1.
        input_dims = [1 for i in range(len(layers))]

        self.encoders = nn.ModuleList([
            GCNN(
                input_dim=input_dims[i],
                hidden_dim=int(self.config["hidden_dim"]),  # int because of pyg
                output_dim=self.config["latent_dim"],
            )
            for i in range(len(layers))
        ])

        # Init output layers
        self.MLPs = nn.ModuleDict()
        for var in self.target_variables:
            if self.dataset.variable_types[var] == "numerical":
                num_class = 1
            else:
                num_class = len(np.unique(self.dataset.ann[var]))
            self.MLPs[var] = MLP(
                input_dim=self.config["latent_dim"] * len(layers),
                hidden_dim=self.config["hidden_dim"],
                output_dim=num_class,
            )

    def forward(self, x_list):
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x.x if x.x.ndim > 1 else x.x.unsqueeze(-1), x.edge_index, x.batch))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings_concat)
        return outputs  

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer

    def compute_loss(self, var, y, y_hat):
        if self.dataset.variable_types[var] == "numerical":
            # Ignore instances with missing labels for numerical variables
            valid_indices = ~torch.isnan(y)
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.mse_loss(torch.flatten(y_hat), y.float())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True) # if no valid labels, set loss to 0
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
            total_loss = sum(torch.exp(-self.log_vars[name]) * loss + self.log_vars[name] for name, loss in losses.items())
        else:
            # Compute unweighted total loss
            total_loss = sum(losses.values())
        return total_loss

    def training_step(self, train_batch, batch_idx):
        dat, y_dict = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]

        outputs = self.forward(x_list)

        losses = {}
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]
            loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = self.compute_total_loss(losses)
        losses["train_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(x_list[0].batch_size))
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        dat, y_dict = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]

        outputs = self.forward(x_list)

        losses = {}
        for var in self.variables:
            y_hat = outputs[var]
            y = y_dict[var]
            loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = sum(losses.values())
        losses["val_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(x_list[0].batch_size))
        return total_loss

    def prepare_data(self):
        lt = int(len(self.dataset) * (1 - self.val_size))
        lv = len(self.dataset) - lt
        dat_train, dat_val = random_split(self.dataset, [lt, lv], generator=torch.Generator().manual_seed(42))
        return dat_train, dat_val

    def train_dataloader(self):
        return DataLoader(self.dat_train, batch_size=int(self.config["batch_size"]), num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dat_val, batch_size=int(self.config["batch_size"]), num_workers=0, pin_memory=True, shuffle=False)

    def predict(self, dataset):
        self.eval()
        layers = dataset.dat.keys()
        x_list = [dataset.dat[x] for x in layers]
        outputs = self.forward(x_list)

        predictions = {}
        for var in self.target_variables:
            y_pred = outputs[var].detach().numpy()
            if self.dataset.variable_types[var] == "categorical":
                predictions[var] = np.argmax(y_pred, axis=1)
            else:
                predictions[var] = y_pred
        return predictions

    def transform(self, dataset):
        self.eval()
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(dataset.dat.values()):
            embeddings_list.append(self.encoders[i](x))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        # Converting tensor to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(
            embeddings_concat.detach().numpy(),
            index=dataset.samples,
            columns=[f"E{dim}" for dim in range(embeddings_concat.shape[1])],
        )
        return embeddings_df

    def forward_target(self, *args):
        """Adaptor forward function for captum integrated gradients.
        """
        raise NotImplementedError

    def compute_feature_importance(self, target_var, steps=5):
        raise NotImplementedError

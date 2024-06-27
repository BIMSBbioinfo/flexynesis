import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import lightning as pl

import numpy as np
import pandas as pd

from captum.attr import IntegratedGradients


from ..modules import CNN


class DirectPredCNN(pl.LightningModule):
    def __init__(self, config, dataset, target_variables, batch_variables=None, val_size=0.2):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.target_variables = target_variables
        self.batch_variables = batch_variables
        self.variables = target_variables + batch_variables if batch_variables else target_variables
        self.val_size = val_size
        self.dat_train, self.dat_val = self.prepare_data()
        self.feature_importances = {}
        # Init modality encoders
        layers = list(self.dataset.dat.keys())
        input_dims = [len(self.dataset.features[layers[i]]) for i in range(len(layers))]
        self.encoders = nn.ModuleList(
            [
                CNN(input_dim=input_dims[i], hidden_dim=self.config["hidden_dim"], output_dim=self.config["latent_dim"])
                for i in range(len(layers))
            ]
        )
        # Init output layers
        layers = list(self.dataset.dat.keys())
        self.MLPs = nn.ModuleDict()
        for var in self.target_variables:
            if self.dataset.variable_types[var] == "numerical":
                num_class = 1
            else:
                num_class = len(np.unique(self.dataset.ann[var]))
            self.MLPs[var] = CNN(
                input_dim=self.config["latent_dim"] * len(layers),
                hidden_dim=self.config["hidden_dim"],
                output_dim=num_class,
            )

    def forward(self, x_list):
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
                loss = torch.tensor(0)  # if no valid labels, set loss to 0
        else:
            # Ignore instances with missing labels for categorical variables
            # Assuming that missing values were encoded as -1
            valid_indices = (y != -1) & (~torch.isnan(y))
            if valid_indices.sum() > 0:  # only calculate loss if there are valid targets
                y_hat = y_hat[valid_indices]
                y = y[valid_indices]
                loss = F.cross_entropy(y_hat, y.long())
            else:
                loss = torch.tensor(0)
        return loss

    def training_step(self, train_batch, batch_idx):
        dat, y_dict = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        outputs = self.forward(x_list)
        losses = {}
        for var in self.target_variables:
            y_hat = outputs[var]
            y = y_dict[var]
            loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss
        total_loss = sum(losses.values())
        losses["train_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        dat, y_dict = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        outputs = self.forward(x_list)
        losses = {}
        for var in self.target_variables:
            y_hat = outputs[var]
            y = y_dict[var]
            loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss
        total_loss = sum(losses.values())
        losses["val_loss"] = total_loss
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def prepare_data(self):
        lt = int(len(self.dataset) * (1 - self.val_size))
        lv = len(self.dataset) - lt
        dat_train, dat_val = random_split(self.dataset, [lt, lv], generator=torch.Generator().manual_seed(42))
        return dat_train, dat_val

    def train_dataloader(self):
        return DataLoader(
            self.dat_train,
            batch_size=int(self.config["batch_size"]),
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dat_val, batch_size=int(self.config["batch_size"]), num_workers=0, pin_memory=True, shuffle=False
        )

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
        return torch.cat(outputs_list, dim=0)

    def compute_feature_importance(self, target_var, steps=5):
        x_list = [self.dataset.dat[x] for x in self.dataset.dat.keys()]

        # Initialize the Integrated Gradients method
        ig = IntegratedGradients(self.forward_target)

        input_data = tuple([data.unsqueeze(0).requires_grad_() for data in x_list])

        # Define a baseline (you might need to adjust this depending on your actual data)
        baseline = tuple([torch.zeros_like(data) for data in input_data])

        # Get the number of classes for the target variable
        if self.dataset.variable_types[target_var] == "numerical":
            num_class = 1
        else:
            num_class = len(np.unique(self.dataset.ann[target_var]))

        # Compute the feature importance for each class
        attributions = []
        if num_class > 1:
            for target_class in range(num_class):
                attributions.append(
                    ig.attribute(
                        input_data,
                        baseline,
                        additional_forward_args=(target_var, steps),
                        target=target_class,
                        n_steps=steps,
                    )
                )
        else:
            attributions.append(
                ig.attribute(input_data, baseline, additional_forward_args=(target_var, steps), n_steps=steps)
            )

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
                df_list.append(
                    pd.DataFrame(
                        {
                            "target_variable": target_var,
                            "target_class": i,
                            "layer": layers[j],
                            "name": features,
                            "importance": importances,
                        }
                    )
                )
        df_imp = pd.concat(df_list, ignore_index=True)

        # save the computed scores in the model
        self.feature_importances[target_var] = df_imp

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split

import lightning as pl

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


from captum.attr import IntegratedGradients

from ..modules import MLP, cox_ph_loss, GNNs


class DirectPredGCNN(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset,
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
        self.variable_types = dataset.variable_types 
        self.ann = dataset.ann 
        
        self.feature_importances = {}
        self.use_loss_weighting = use_loss_weighting

        self.device_type = device_type 
        self.gnn_conv_type = gnn_conv_type
                
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for var in self.variables:
                self.log_vars[var] = nn.Parameter(torch.zeros(1))

        # Init modality encoders
        self.layers = list(dataset.dat.keys())
        # NOTE: For now we use matrices, so number of node input features is 1.
        input_dims = [1 for _ in range(len(self.layers))]
        
        # define this to be able to make hidden_dim as a factor of number of features in each layer
        feature_counts = [len(dataset.features[x]) for x in self.layers]
        
        self.encoders = nn.ModuleList(
            [
                GNNs(
                        input_dim=input_dims[i],
                        hidden_dim=int(self.config["hidden_dim_factor"] * feature_counts[i]),  
                        output_dim=self.config["latent_dim"],
                        act = self.config['activation'],
                        conv = self.gnn_conv_type
                    )
                    for i in range(len(self.layers))       
            ])

        # Init output layers
        self.MLPs = nn.ModuleDict()
        for var in self.variables:
            if self.variable_types[var] == "numerical":
                num_class = 1
            else:
                num_class = len(np.unique(self.ann[var]))
            self.MLPs[var] = MLP(
                input_dim=self.config["latent_dim"] * len(self.layers),
                hidden_dim=self.config["supervisor_hidden_dim"],
                output_dim=num_class,
            )

    def forward(self, x_list):
        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x.x, x.edge_index, x.batch))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings_concat)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
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

    def training_step(self, train_batch):
        (dat, y_dict), idx = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]

        outputs = self.forward(x_list)

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
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(x_list[0].batch_size))
        return total_loss

    def validation_step(self, val_batch):
        (dat, y_dict), idx = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]

        outputs = self.forward(x_list)

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
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(x_list[0].batch_size))
        return total_loss


    def predict(self, dataset):
        self.eval()
        xs = [x for x in dataset.dat.values()]
        edge_indices = [dataset.feature_ann[k]["edge_index"] for k in self.layers]
        inputs = []
        for x, edge_idx in zip(xs, edge_indices):
            inputs.append(
                Batch.from_data_list(
                    [Data(x=sample.unsqueeze(1) if sample.ndim == 1 else sample, edge_index=edge_idx) for sample in x]
                )
            )
        outputs = self.forward(inputs)

        predictions = {}
        for var in self.variables:
            y_pred = outputs[var].detach().numpy()
            if self.variable_types[var] == "categorical":
                predictions[var] = np.argmax(y_pred, axis=1)
            else:
                predictions[var] = y_pred
        return predictions

    def transform(self, dataset):
        self.eval()

        xs = [x for x in dataset.dat.values()]
        edge_indices = [dataset.feature_ann[k]["edge_index"] for k in self.layers]
        inputs = []
        for x, edge_idx in zip(xs, edge_indices):
            inputs.append(
                Batch.from_data_list(
                    [Data(x=sample.unsqueeze(1) if sample.ndim == 1 else sample, edge_index=edge_idx) for sample in x]
                )
            )

        embeddings_list = []
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(inputs):
            embeddings_list.append(self.encoders[i](x.x, x.edge_index, x.batch))
        embeddings_concat = torch.cat(embeddings_list, dim=1)

        # Converting tensor to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(
            embeddings_concat.detach().numpy(),
            index=dataset.samples,
            columns=[f"E{dim}" for dim in range(embeddings_concat.shape[1])],
        )
        return embeddings_df

    def model_forward(self, *args):
        xs = list(args[:-2])
        edge_index_arg = args[-2]
        target_var = args[-1]
        inputs = []
        for x, edge_idx in zip(xs, edge_index_arg):
            inputs.append(
                Batch.from_data_list(
                    [Data(x=sample.unsqueeze(1) if sample.ndim == 1 else sample, edge_index=edge_idx) for sample in x]
                )
            )
        return self.forward(inputs)[target_var]

    def compute_feature_importance(self, dataset, target_var, steps=5, batch_size = 32):
        # find out the device the model was trained on.
        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.to(device)

        def bytes_to_gb(bytes):
            return bytes / 1024 ** 2
        
        print("[INFO] Computing feature importance for variable:",target_var,"on device:",device)

        # notice the DataLoader comes from torch_geometric.loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        ig = IntegratedGradients(self.model_forward)

        # Get the number of classes for the target variable
        if dataset.variable_types[target_var] == "numerical":
            num_class = 1
        else:
            num_class = len(np.unique(dataset.ann[target_var]))

        aggregated_attributions = [[] for _ in range(num_class)]
        
        print("Memory before batch processing: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))
        for batch in dataloader:
            (dat, ydict), idx = batch
            print("processing ",len(idx),"samples in batch")
            subset = dataset[idx]
            
            # Prepare inputs and baselines, moving them to the GPU
            xs = [x.to(device) for x in subset.dat.values()]
            
            print("Memory after moving xs to GPU: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))
            edge_indices = [subset.feature_ann[k]["edge_index"].to(device) for k in subset.dat.keys()]
            print("Memory after moving edges to GPU: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))
            
            inputs = tuple(xs)
            baselines = tuple([torch.zeros_like(x) for x in inputs])

            edge_index_arg = tuple(edge_indices)
            additional_forward_args = (edge_index_arg, target_var)
            
            del xs  # Explicitly delete if not needed anymore
            torch.cuda.empty_cache()
            print("Memory after clearing cache: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))

            
            #internal_batch_size = inputs[0].shape[0]
            #print("Internal batch size:",internal_batch_size)
            if num_class == 1:
                # returns a tuple of tensors (one per data modality)
                attributions = ig.attribute(inputs, baselines,
                                            additional_forward_args=additional_forward_args,
                                            n_steps=steps) 
                                            #internal_batch_size = internal_batch_size)  
                aggregated_attributions[0].append(attributions)
            else:
                for target_class in range(num_class):
                    attributions = ig.attribute(
                            inputs,
                            baselines,
                            additional_forward_args=additional_forward_args,
                            target=target_class,
                            n_steps=steps) 
                            #internal_batch_size = internal_batch_size)
                    aggregated_attributions[target_class].append(attributions)

        # For each target class and for each data modality/layer, concatenate attributions accross batches 
        layers = list(dataset.dat.keys())
        num_layers = len(layers)
        processed_attributions = [] 
        # Process each class
        for class_idx in range(len(aggregated_attributions)):
            class_attr = aggregated_attributions[class_idx]
            layer_attributions = []
            # Process each layer within the class
            for layer_idx in range(num_layers):
                # Extract all batch tensors for this layer across all batches for the current class
                layer_tensors = [batch_attr[layer_idx] for batch_attr in class_attr]
                # Concatenate tensors along the batch dimension
                attr_concat = torch.cat(layer_tensors, dim=1)
                layer_attributions.append(attr_concat)
            processed_attributions.append(layer_attributions)

        # compute absolute importance and move to cpu 
        abs_attr = [[torch.abs(a).cpu() for a in attr_class] for attr_class in processed_attributions]
        # average over samples 
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]
        # move the model also back to cpu (if not already on cpu)
        self.to('cpu')

        # combine into a single data frame
        df_list = []
        for i in range(num_class):
            for j in range(len(layers)):
                features = dataset.features[layers[j]]
                # Ensure tensors are already on CPU before converting to numpy
                importances = imp[i][j][0].detach().numpy()
                df_list.append(pd.DataFrame({'target_variable': target_var, 
                                             'target_class': i, 
                                             'layer': layers[j], 
                                             'name': features, 
                                             'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index=True)
        # Save the computed scores in the model
        self.feature_importances[target_var] = df_imp

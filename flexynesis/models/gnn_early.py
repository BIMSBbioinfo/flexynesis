import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split

import lightning as pl

from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients

from ..modules import MLP, cox_ph_loss, flexGCN


class GNN(pl.LightningModule):
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
        
        self.encoders = flexGCN(
                        node_count = dataset[0][0].shape[0], #number of nodes
                        node_feature_count= dataset[0][0].shape[1], # number of node features
                        node_embedding_dim=int(self.config["node_embedding_dim"]),  
                        num_convs = int(self.config['num_convs']), # Number of convolutional layers 
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
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
        edge_index = dataset.edge_index.to(device)  # Move edge_index to GPU

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust the batch size as needed
        all_embeddings = []  # List to store embeddings from all batches
        sample_ids = []  # List to store indices for all samples processed

        # Process each batch
        for x, y_dict, samples in dataloader:
            x = x.to(device)  # Move data to GPU

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
        
    # Adaptor forward function for captum integrated gradients. 
    def forward_target(self, *args):
        input_data = list(args[:-2])  # expect a single tensor (early integration)
        target_var = args[-2]  # target variable of interest
        steps = args[-1]  # number of steps for IntegratedGradients().attribute 
        outputs_list = []
        for i in range(steps):
            x_step = input_data[0][i] 
            #edges_step = edge_index[i] # although, identical, they get copied. 
            out = self.forward(x_step, self.dataset_edge_index)
            outputs_list.append(out[target_var])
        return torch.cat(outputs_list, dim = 0)

        
    def compute_feature_importance(self, dataset, target_var, steps=5, batch_size = 32):
        """
        Computes the feature importance for each variable in the dataset using the Integrated Gradients method.
        This method measures the importance of each feature by attributing the prediction output to each input feature.
    
        Args:
            dataset: The dataset object containing the features and data (MultiOmicDatasetNW object).
            target_var (str): The target variable for which feature importance is calculated.
            steps (int, optional): The number of steps to use for integrated gradients approximation. Defaults to 5.
            batch_size (int, optional): The size of the batch to process the dataset. Defaults to 64.
    
        Returns:
            pd.DataFrame: A DataFrame containing feature importances across different variables and data modalities.
                          Columns include 'target_variable', 'target_class', 'target_class_label', 'layer', 'name',
                          and 'importance'.
    
        This function adjusts the device setting based on the availability of GPUs and performs the computation using
        Integrated Gradients. It processes batches of data, aggregates results across batches, and formats the output
        into a readable DataFrame which is then stored in the model's attribute for later use or analysis.
        """
        def bytes_to_gb(bytes):
            return bytes / 1024 ** 2
        print("Memory before moving model to device: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))
        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.to(device)
        print("Memory before edges: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))
        self.dataset_edge_index = dataset.edge_index.to(device)
        print("Memory after edges: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))


        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ig = IntegratedGradients(self.forward_target)

        if dataset.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique(dataset.ann[target_var]))
        
        print("Memory before batch processing: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))

        aggregated_attributions = [[] for _ in range(num_class)]
        for batch in dataloader:
            x, y_dict, samples = batch
            
            input_data = x.unsqueeze(0).requires_grad_().to(device)
            baseline = torch.zeros_like(input_data)
            
            if num_class == 1:
                # returns a tuple of tensors (one per data modality)
                attributions = ig.attribute( input_data, baseline, 
                                             additional_forward_args=(target_var, steps), 
                                             n_steps=steps)
                aggregated_attributions[0].append(attributions)
            else:
                for target_class in range(num_class):
                    # returns a tuple of tensors (one per data modality)
                    attributions = ig.attribute( input_data, baseline, 
                                                 additional_forward_args=(target_var, steps), 
                                                 target=target_class, n_steps=steps)
                    aggregated_attributions[target_class].append(attributions)
        # For each target class concatenate node attributions accross batches 
        processed_attributions = [] 
        # Process each class
        for class_idx in range(len(aggregated_attributions)):
            class_attr = aggregated_attributions[class_idx]
            # Concatenate tensors along the batch dimension
            attr_concat = torch.cat([batch_attr for batch_attr in class_attr], dim=1)
            processed_attributions.append(attr_concat)

        # compute absolute importance and move to cpu 
        abs_attr = [torch.abs(attr_class).cpu() for attr_class in processed_attributions]
        # average over samples 
        imp = [a.mean(dim=1) for a in abs_attr]

        # move the model also back to cpu (if not already on cpu)
        self.to('cpu')
        print("Memory after batch processing: {:.3f} MB".format(bytes_to_gb(torch.cuda.max_memory_reserved())))


        df_list = []
        layers = list(dataset.multiomic_dataset.dat.keys())
        for i in range(num_class):
            features = dataset.common_features
            target_class_label = dataset.label_mappings[target_var].get(i) if target_var in dataset.label_mappings else ''
            for l in range(len(layers)): 
                # extracting node feature attributes coming from different omic layers 
                importances = imp[i].squeeze().detach().numpy()[:,l]
                df_list.append(pd.DataFrame({'target_variable': target_var, 
                                             'target_class': i, 
                                             'target_class_label': target_class_label,
                                             'layer': layers[l], 
                                             'name': features, 
                                             'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index=True)
        # save the computed scores in the model
        self.feature_importances[target_var] = df_imp
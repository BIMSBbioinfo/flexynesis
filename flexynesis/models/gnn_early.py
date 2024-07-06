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
    """
    A Graph Neural Network module implemented with PyTorch Lightning, designed for
    multi-omic data.

    This class integrates various graph convolution types and supports complex tasks
    such as single/multi-task regression/classification/survival prediction.
    It allows for extensive configuration including device type and convolution type,
    and dynamically constructs output layers based on the variable types provided.

    Attributes:
        config (dict): Configuration dictionary specifying model parameters like
                       node embedding dimensions, number of convolutions, activation functions, etc.
        target_variables (list): List of target variables for prediction.
        surv_event_var (str, optional): Name of the survival event variable if survival analysis is performed.
        surv_time_var (str, optional): Name of the survival time variable if survival analysis is performed.
        batch_variables (list, optional): List of batch variables for handling batch effects.
        use_loss_weighting (bool): If True, uses log variance for uncertainty weighting in loss calculations.
        device_type (str, optional): Device type to use ('gpu' or 'cpu'). Defaults to 'cpu' if not specified.
        gnn_conv_type (str, optional): Type of graph convolutional layer to use (options: GC, SAGE, GCN)
        variable_types (dict): Dictionary mapping variables to their types (e.g., numerical, categorical).
        ann (DataFrame): Annotation DataFrame from the dataset containing variables and their annotations.
        edge_index (Tensor): Tensor describing the edge connections in the graph.
        feature_importances (dict): Dictionary to store feature importances if computed.
        encoders (Module): Graph convolutional network module for feature encoding.
        MLPs (ModuleDict): Dictionary of multi-layer perceptrons, one for each target variable.

    Args:
        config (dict): Configuration settings for model parameters.
        dataset (MultiOmicGeometricDataset): Dataset object containing graph data and annotations.
        target_variables (list): Names of the variables to be predicted.
        batch_variables (list, optional): Names of the variables that represent batch effects.
        surv_event_var (str, optional): The variable name representing survival events.
        surv_time_var (str, optional): The variable name representing survival times.
        use_loss_weighting (bool, optional): Whether to use uncertainty weighting in loss calculation. Defaults to True.
        device_type (str, optional): Specifies the computation device ('gpu' or 'cpu'). Default is None, which uses 'cpu' if 'gpu' is not available.
        gnn_conv_type (str, optional): Specifies the type of graph convolutional layer to use.
    """    
    def __init__(
        self,
        config,
        dataset, # MultiomicDatasetNW object
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
        """
        Defines the forward pass of the GNN.

        Args:
            x (Tensor): Node feature matrix (batch_size, num_nodes, node_feature_count).
            edge_index (Tensor): Edge index in COO format (2, num_edges).

        Returns:
            dict: Outputs from the MLPs, one for each target variable.
        """
        embeddings = self.encoders(x, edge_index)
        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(embeddings)
        return outputs

            
    def training_step(self, batch):
        """
        Performs a training step including loss calculation and logging.

        Args:
            batch (tuple): A batch of data consisting of features, target labels as a dictionary of tensors, and sample ids.

        Returns:
            float: Total loss for the batch.
        """
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
        """
        Performs a validation step, computing losses for a batch of data.

        Args:
            batch (tuple): A batch of data consisting of features, target labels as a dictionary of tensors, and sample ids.

        Returns:
            float: Total validation loss for the batch.
        """
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
        """
        Configure the optimizer for the DirectPred model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def compute_loss(self, var, y, y_hat):
        """
        Computes the loss for a specific variable based on whether the variable is numerical or categorical.
        Handles missing labels by excluding them from the loss calculation.
    
        Args:
            var (str): The name of the variable for which the loss is being calculated.
            y (torch.Tensor): The true labels or values for the variable.
            y_hat (torch.Tensor): The predicted labels or values output by the model.
    
        Returns:
            torch.Tensor: The calculated loss tensor for the variable. If there are no valid labels or values
                          to compute the loss (all are missing), returns a zero loss tensor with gradient enabled.
    
        The method first checks the type of the variable (`var`) from `variable_types`. If the variable is
        numerical, it computes the mean squared error loss. For categorical variables, it calculates the
        cross-entropy loss. The method ensures to ignore any instances where the labels are missing (NaN for
        numerical or -1 for categorical as assumed missing value encoding) when calculating the loss.
        """
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
        """
        Computes the total loss from a dictionary of individual losses. This method can compute
        either weighted or unweighted total loss based on the model configuration. If loss weighting
        is enabled and there are multiple loss components, it uses uncertainty-based weighting.
        See Kendall A. et al, https://arxiv.org/abs/1705.07115.
        
        Args:
            losses (dict of torch.Tensor): A dictionary where each key is a variable name and
                                           each value is the loss tensor associated with that variable.
    
        Returns:
            torch.Tensor: The total loss computed across all inputs, either weighted or unweighted.
        
        The method checks if loss weighting is used (`use_loss_weighting`) and if there are multiple
        losses to weight. If so, it computes the weighted sum of losses, where the weight involves
        the exponential of the negative log variance (acting as precision) associated with each loss,
        added to the log variance itself. This approach helps in balancing the contribution of each
        loss component based on its uncertainty. If loss weighting is not used, or there is only one
        loss component, it sums up the losses directly.
        """
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
        """
        Make predictions on an entire dataset.

        Args:
            dataset: The MultiOmicDatasetNW object to evaluate the model on.

        Returns:
            dict: Predictions mapped by target variable names.
        """
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
        """
        Transforms the input data into a lower-dimensional representation using trained encoders.

        Args:
            dataset: The MultiOmicDatasetNW containing the input data.

        Returns:
            pd.DataFrame: DataFrame containing the transformed data.
        """
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
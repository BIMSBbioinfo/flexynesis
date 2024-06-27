# Generating encodings of multi-omic data using triplet loss 
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import itertools

import pandas as pd
import numpy as np

import lightning as pl

from ..modules import *
from ..data import TripletMultiOmicDataset

from captum.attr import IntegratedGradients



class MultiTripletNetwork(pl.LightningModule):
    """
    A PyTorch Lightning module that implements a multi-triplet network architecture for handling
    both categorical and numerical target variables,using triplet loss for learning
    discriminative embeddings. The network can also handle survival analysis variables and implements
    loss weighting to manage uncertainty.

    Attributes:
        config (dict): Configuration settings for model parameters such as learning rates, dimensions, etc.
        dataset: The dataset object that contains the data and metadata.
        target_variables (list): A list of the names of the target variables that the model will predict.
        batch_variables (list, optional): A list of batch variable names used for batch effect correction.
        surv_event_var (str, optional): The name of the survival event variable.
        surv_time_var (str, optional): The name of the survival time variable.
        use_loss_weighting (bool): Flag indicating whether loss weighting is used to handle uncertainty.
        device_type (str, optional): Type of device ('gpu' or 'cpu') on which the model will be run.
    """

    def __init__(self, config, dataset, target_variables, batch_variables = None, 
                 surv_event_var = None, surv_time_var = None, use_loss_weighting = True, 
                 device_type = None):
        super(MultiTripletNetwork, self).__init__()
        
        self.config = config
        self.target_variables = target_variables
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        # both surv event and time variables are assumed to be numerical variables
        # we create only one survival variable for the pair (surv_time_var and surv_event_var)
        if self.surv_event_var is not None and self.surv_time_var is not None:
            self.target_variables = self.target_variables + [self.surv_event_var]
        self.batch_variables = batch_variables
        self.variables = self.target_variables + batch_variables if batch_variables else self.target_variables
        self.ann = dataset.ann
        self.variable_types = dataset.variable_types
        self.feature_importances = {}
        self.device_type = device_type
        # The first target variable is the main variable that dictates the triplets 
        # it has to be a categorical variable 
        self.main_var = self.target_variables[0] 
        if self.variable_types[self.main_var] == 'numerical':
            raise ValueError("The first target variable",self.main_var," must be a categorical variable")
        
        self.use_loss_weighting = use_loss_weighting
        
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for loss_type in itertools.chain(self.variables, ['triplet_loss']):
                self.log_vars[loss_type] = nn.Parameter(torch.zeros(1))
        
        self.layers = list(dataset.dat.keys())
        self.input_dims = [len(dataset.features[self.layers[i]]) for i in range(len(self.layers))]

        self.encoders = nn.ModuleList([
            MLP(input_dim=self.input_dims[i],
                # define hidden_dim size relative to the input_dim size
                hidden_dim=int(self.input_dims[i] * self.config['hidden_dim_factor']),
                output_dim=self.config['latent_dim']) for i in range(len(self.layers))])
        
        # define supervisor heads for both target and batch variables 
        self.MLPs = nn.ModuleDict() # using ModuleDict to store multiple MLPs
        for var in self.variables:
            if self.variable_types[var] == 'numerical':
                num_class = 1
            else:
                num_class = len(np.unique(self.ann[var]))
            self.MLPs[var] = MLP(input_dim=self.config['latent_dim'] * len(self.layers),
                                 hidden_dim=self.config['supervisor_hidden_dim'],
                                 output_dim=num_class)
    
    def concat_embeddings(self, dat):
        embeddings_list = []
        x_list = [dat[x] for x in dat.keys()]
        # Process each input matrix with its corresponding Encoder
        for i, x in enumerate(x_list):
            embeddings_list.append(self.encoders[i](x))
        embeddings_concat = torch.cat(embeddings_list, dim=1)
        return embeddings_concat
        
    def forward(self, anchor, positive, negative):
        """
        Compute the forward pass of the MultiTripletNetwork and return the embeddings and predictions.

        Args:
            anchor (dict): A dictionary containing the anchor input tensors
            positive (dict): A dictionary containing the positive input tensors
            negative (dict): A dictionary containing the negative input tensors

        Returns:
            tuple: A tuple containing the anchor, positive, and negative embeddings and the predicted class labels.
        """
        # triplet encoding
        anchor_embedding = self.concat_embeddings(anchor)
        positive_embedding = self.concat_embeddings(positive)
        negative_embedding = self.concat_embeddings(negative)
        
        #run the supervisor heads using the anchor embeddings as input
        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(anchor_embedding)
        return anchor_embedding, positive_embedding, negative_embedding, outputs
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the MultiTripletNetwork.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        """
        Compute the triplet loss for the given anchor, positive, and negative embeddings.

        Args:
            anchor (torch.Tensor): The anchor embedding tensor.
            positive (torch.Tensor): The positive embedding tensor.
            negative (torch.Tensor): The negative embedding tensor.
            margin (float, optional): The margin for the triplet loss. Default is 1.0.

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()    

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
        if self.variable_types[var] == 'numerical':
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
            total_loss = sum(torch.exp(-self.log_vars[name]) * loss + self.log_vars[name] for name, loss in losses.items())
        else:
            # Compute unweighted total loss
            total_loss = sum(losses.values())
        return total_loss

    def training_step(self, train_batch, batch_idx, log = True):
        """
        Perform a training step using a single batch of data, including triplet components and target labels.
    
        Args:
            train_batch (tuple): The batch containing data tuples (anchor, positive, negative) and a dictionary of labels.
            batch_idx (int): The index of the current batch.
            log (bool, optional): Flag to determine if logging should occur at each step. Defaults to True.
    
        Returns:
            torch.Tensor: The total loss for the current training batch, which includes triplet loss and any additional
                          losses from supervisor heads.
    
        This method computes the embedding for the anchor, positive, and negative samples and calculates the triplet loss.
        Additional losses are computed for other target variables in the dataset, particularly handling survival analysis
        if applicable. All losses are combined to compute a total loss, which is logged and returned.
        """
        anchor, positive, negative, y_dict = train_batch[0], train_batch[1], train_batch[2], train_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, outputs = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        
        # compute loss values for the supervisor heads 
        losses = {'triplet_loss': triplet_loss}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var] 
                risk_scores = outputs[var] #output of MLP
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = self.compute_total_loss(losses)
        # add total loss for logging 
        losses['train_loss'] = total_loss
        if log:
            self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
    
    def validation_step(self, val_batch, batch_idx, log = True):
        """
        Perform a validation step using a single batch of data, including triplet components and target labels.
    
        Args:
            val_batch (tuple): The batch containing data tuples (anchor, positive, negative) and a dictionary of labels.
            batch_idx (int): The index of the current batch.
            log (bool, optional): Flag to determine if logging should occur at each step. Defaults to True.
    
        Returns:
            torch.Tensor: The total loss for the current validation batch, which includes triplet loss and any additional
                          losses from supervisor heads.
    
        Similar to the training step, this method computes the embedding for the anchor, positive, and negative samples
        and calculates the triplet loss. It computes additional losses for other target variables in the dataset, aggregates
        all losses, and returns the total loss. The losses are logged if specified.
        """
        anchor, positive, negative, y_dict = val_batch[0], val_batch[1], val_batch[2], val_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, outputs = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        
        # compute loss values for the supervisor heads 
        losses = {'triplet_loss': triplet_loss}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var] 
                risk_scores = outputs[var] #output of MLP
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss
        
        total_loss = sum(losses.values())
        losses['val_loss'] = total_loss
        if log:
            self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
        
    # dataset: MultiOmicDataset
    def transform(self, dataset):
        """
        Transforms the input dataset by generating embeddings and predictions.
        
        Args:
            dataset (MultiOmicDataset): An instance of the MultiOmicDataset class.
            
        Returns:
            z (pd.DataFrame): A dataframe containing the computed embeddings.
            y_pred (np.ndarray): A numpy array containing the predicted labels.
        """
        self.eval()
        # get anchor embeddings 
        z = pd.DataFrame(self.concat_embeddings(dataset.dat).detach().numpy())
        z.columns = [''.join(['E', str(x)]) for x in z.columns]
        z.index = dataset.samples
        return z

    def predict(self, dataset):
        """
        Evaluate the model on a given dataset.

        Args:
            dataset: The dataset to evaluate the model on.

        Returns:
            A dictionary where each key is a target variable and the corresponding value is the predicted output for that variable.
        """
        self.eval()
        # get anchor embedding
        anchor_embedding = self.concat_embeddings(dataset.dat)
        # get MLP outputs for each var
        outputs = {}
        for var, mlp in self.MLPs.items():
            outputs[var] = mlp(anchor_embedding)
        
        # get predictions from the mlp outputs for each var
        predictions = {}
        for var in self.variables:
            y_pred = outputs[var].detach().numpy()
            if self.variable_types[var] == 'categorical':
                predictions[var] = np.argmax(y_pred, axis=1)
            else:
                predictions[var] = y_pred
        return predictions

    
    # Adaptor forward function for captum integrated gradients. 
    # layer_sizes: number of features in each omic layer 
    def forward_target(self, input_data, layer_sizes, target_var, steps):
        outputs_list = []
        for i in range(steps):
            # for each step, get anchor/positive/negative tensors 
            # (split the concatenated omics layers)
            anchor = input_data[i][0].split(layer_sizes, dim = 1)
            positive = input_data[i][1].split(layer_sizes, dim = 1)
            negative = input_data[i][2].split(layer_sizes, dim = 1)
            
            # convert to dict
            anchor = {k: anchor[k] for k in range(len(anchor))}
            positive = {k: anchor[k] for k in range(len(positive))}
            negative = {k: anchor[k] for k in range(len(negative))}
            anchor_embedding, positive_embedding, negative_embedding, outputs = self.forward(anchor, positive, negative)
            outputs_list.append(outputs[target_var])
        return torch.cat(outputs_list, dim = 0)
        
    def compute_feature_importance(self, dataset, target_var, steps = 5, batch_size = 64):
        """
        Computes the feature importance for each variable in the dataset using the Integrated Gradients method.
        This method measures the importance of each feature by attributing the prediction output to each input feature.
    
        Args:
            dataset: The dataset object containing the features and data.
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

        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.to(device)

        print("[INFO] Computing feature importance for variable:",target_var,"on device:",device)

        # define data loader 
        triplet_dataset = TripletMultiOmicDataset(dataset, self.main_var)
        dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=False)
        # Initialize the Integrated Gradients method
        ig = IntegratedGradients(self.forward_target)

        if self.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique([y[target_var] for _, y in dataset]))

        aggregated_attributions = [[] for _ in range(num_class)]
        for batch in dataloader:
            # see training_step to see how elements are accessed in batches 
            anchor, positive, negative, y_dict = batch[0], batch[1], batch[2], batch[3]        

            # Move tensors to the specified device
            anchor = {k: v.to(device) for k, v in anchor.items()}
            positive = {k: v.to(device) for k, v in positive.items()}
            negative = {k: v.to(device) for k, v in negative.items()}
            
            anchor = [data.requires_grad_() for data in list(anchor.values())]
            positive = [data.requires_grad_() for data in list(positive.values())]
            negative = [data.requires_grad_() for data in list(negative.values())]
            
            # concatenate multiomic layers of each list element
            # then stack the anchor/positive/negative 
            # the purpose is to get a single tensor
            input_data = torch.stack([torch.cat(sublist, dim = 1) for sublist in [anchor, positive, negative]]).unsqueeze(0)

            # layer sizes will be needed to revert the concatenated tensor 
            # anchor/positive/negative have the same shape
            layer_sizes = [anchor[i].shape[1] for i in range(len(anchor))] 

            # Define a baseline 
            baseline = torch.zeros_like(input_data)       

            if num_class == 1:
                # returns a tuple of tensors (one per data modality)
                attributions = ig.attribute(input_data, baseline, 
                                             additional_forward_args=(layer_sizes, target_var, steps), 
                                             n_steps=steps)
                attributions = attributions.split(layer_sizes, dim = 3)
                aggregated_attributions[0].append(attributions)
            else:
                for target_class in range(num_class):
                    # returns a tuple of tensors (one per data modality)
                    attributions = ig.attribute(input_data, baseline, 
                                                 additional_forward_args=(layer_sizes, target_var, steps), 
                                                 target=target_class, n_steps=steps)
                    attributions = attributions.split(layer_sizes, dim = 3)
                    aggregated_attributions[target_class].append(attributions)

        # For each target class and for each data modality/layer, concatenate attributions accross batches 
        layers = self.layers
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
                attr_concat = torch.cat(layer_tensors, dim=2)
                layer_attributions.append(attr_concat)
            processed_attributions.append(layer_attributions)

        # compute absolute importance and move to cpu 
        # notice the squeeze (due to triplets)
        abs_attr = [[torch.abs(a.squeeze()).cpu() for a in attr_class] for attr_class in processed_attributions]
        # average over samples 
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]
        # move the model also back to cpu (if not already on cpu)
        self.to('cpu')
        df_list = []
        for i in range(num_class):
            for j in range(len(layers)):
                features = dataset.features[layers[j]]
                # Ensure tensors are already on CPU before converting to numpy
                importances = imp[i][j][0].detach().numpy() # 0 => extract importances only for the anchor
                target_class_label = dataset.label_mappings[target_var].get(i) if target_var in dataset.label_mappings else ''
                df_list.append(pd.DataFrame({'target_variable': target_var, 
                                             'target_class': i, 
                                             'target_class_label': target_class_label,
                                             'layer': layers[j], 
                                             'name': features, 
                                             'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index=True)
        self.feature_importances[target_var] = df_imp

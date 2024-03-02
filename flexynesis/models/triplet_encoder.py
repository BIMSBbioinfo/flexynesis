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


class MultiEmbeddingNetwork(nn.Module):
    """
    A neural network module that computes multiple embeddings and fuses them into a single representation.

    Attributes:
        embedding_networks (nn.ModuleList): A list of EmbeddingNetwork instances for each input feature.
    """
    def __init__(self, input_sizes, hidden_sizes, output_size):
        """
        Initialize the MultiEmbeddingNetwork with the given input sizes, hidden sizes, and output size.

        Args:
            input_sizes (list of int): A list of input sizes for each EmbeddingNetwork instance.
            hidden_sizes (list of int): A list of hidden sizes for each EmbeddingNetwork instance.
            output_size (int): The size of the fused embedding output.
        """
        super(MultiEmbeddingNetwork, self).__init__()
        self.embedding_networks = nn.ModuleList([
            EmbeddingNetwork(input_size, hidden_size, output_size)
            for input_size, hidden_size in zip(input_sizes, hidden_sizes)
        ])

    def forward(self, x):
        """
        Compute the forward pass of the MultiEmbeddingNetwork and return the fused embedding.

        Args:
            x (dict): A dictionary containing the input tensors for each EmbeddingNetwork.
                      Keys should correspond to the feature names and values to the input tensors.
        
        Returns:
            torch.Tensor: The fused embedding tensor resulting from the concatenation of individual embeddings.
        """
        embeddings = [
            embedding_network(x[key])
            for key, embedding_network in zip(x.keys(), self.embedding_networks)
        ]
        fused_embedding = torch.cat((embeddings), dim=-1)
        return fused_embedding

class MultiTripletNetwork(pl.LightningModule):
    """
    """
    def __init__(self, config, dataset, target_variables, batch_variables = None, 
                 surv_event_var = None, surv_time_var = None, val_size = 0.2, use_loss_weighting = True, 
                 device_type = None):
        """
        Initialize the MultiTripletNetwork with the given parameters.

        Args:
            TODO
        """
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
        self.val_size = val_size
        self.dataset = dataset
        self.ann = self.dataset.ann
        self.variable_types = self.dataset.variable_types
        self.feature_importances = {}
        self.device_type = device_type

        layers = list(dataset.dat.keys())
        input_sizes = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        hidden_sizes = [config['hidden_dim'] for x in range(len(layers))]
        
        
        # The first target variable is the main variable that dictates the triplets 
        # it has to be a categorical variable 
        main_var = self.target_variables[0] 
        if self.dataset.variable_types[main_var] == 'numerical':
            raise ValueError("The first target variable",main_var," must be a categorical variable")
        
        self.use_loss_weighting = use_loss_weighting
        
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for loss_type in itertools.chain(self.variables, ['triplet_loss']):
                self.log_vars[loss_type] = nn.Parameter(torch.zeros(1))
        
        
        # create train/validation splits and convert TripletMultiOmicDataset format
        self.dataset = TripletMultiOmicDataset(self.dataset, main_var)
        self.dat_train, self.dat_val = self.prepare_data() 
        
        # define embedding network for data matrices 
        self.multi_embedding_network = MultiEmbeddingNetwork(input_sizes, hidden_sizes, config['latent_dim'])

        # define supervisor heads for both target and batch variables 
        self.MLPs = nn.ModuleDict() # using ModuleDict to store multiple MLPs
        for var in self.variables:
            if self.variable_types[var] == 'numerical':
                num_class = 1
            else:
                num_class = len(np.unique(self.ann[var]))
            self.MLPs[var] = MLP(input_dim=self.config['latent_dim'] * len(layers),
                                 hidden_dim=self.config['supervisor_hidden_dim'],
                                 output_dim=num_class)
                                                                              
    def forward(self, anchor, positive, negative):
        """
        Compute the forward pass of the MultiTripletNetwork and return the embeddings and predictions.

        Args:
            anchor (dict): A dictionary containing the anchor input tensors for each EmbeddingNetwork.
            positive (dict): A dictionary containing the positive input tensors for each EmbeddingNetwork.
            negative (dict): A dictionary containing the negative input tensors for each EmbeddingNetwork.

        Returns:
            tuple: A tuple containing the anchor, positive, and negative embeddings and the predicted class labels.
        """
        # triplet encoding
        anchor_embedding = self.multi_embedding_network(anchor)
        positive_embedding = self.multi_embedding_network(positive)
        negative_embedding = self.multi_embedding_network(negative)
        
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
        if self.use_loss_weighting and len(losses) > 1:
            # Compute weighted loss for each loss 
            # Weighted loss = precision * loss + log-variance
            total_loss = sum(torch.exp(-self.log_vars[name]) * loss + self.log_vars[name] for name, loss in losses.items())
        else:
            # Compute unweighted total loss
            total_loss = sum(losses.values())
        return total_loss

    def training_step(self, train_batch, batch_idx):
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
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
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
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
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
        z = pd.DataFrame(self.multi_embedding_network(dataset.dat).detach().numpy())
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
        anchor_embedding = self.multi_embedding_network(dataset.dat)
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
        
    def compute_feature_importance(self, target_var, steps = 5):
        """
        Compute the feature importance.

        Args:
            input_data (torch.Tensor): The input data to compute the feature importance for.
            target_var (str): The target variable to compute the feature importance for.
        Returns:
            attributions (list of torch.Tensor): The feature importances for each class.
        """
        
        device = torch.device("cuda" if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        print("[INFO] Computing feature importance for variable:",target_var,"on device:",device)

        # self.dataset is a TripletMultiomicDataset, which has a different 
        # structure than the MultiomicDataset. We use data loader to 
        # read the triplets and get anchor/positive/negative tensors
        # read the whole dataset
        dl = DataLoader(self.dataset, batch_size=len(self.dataset))
        it = iter(dl)
        anchor, positive, negative, y_dict = next(it) 
                
        # Move tensors to the specified device
        anchor = {k: v.to(device) for k, v in anchor.items()}
        positive = {k: v.to(device) for k, v in positive.items()}
        negative = {k: v.to(device) for k, v in negative.items()}
        y_dict = {k: v.to(device) for k, v in y_dict.items()}
                
        # Initialize the Integrated Gradients method
        ig = IntegratedGradients(self.forward_target)

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

        # Get the number of classes for the target variable
        if self.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique(self.ann[target_var]))

        # Compute the feature importance for each class
        attributions = []
        if num_class > 1:
            for target_class in range(num_class):
                attributions.append(ig.attribute(input_data, baseline, additional_forward_args=(layer_sizes, target_var, steps), target=target_class, n_steps=steps))
        else:
            attributions.append(ig.attribute(input_data, baseline, additional_forward_args=(layer_sizes, target_var, steps), n_steps=steps))

        # summarize feature importances
        # Compute absolute attributions
        # Move the processed tensors to CPU for further operations that are not supported on GPU
        abs_attr = [[torch.abs(a).cpu() for a in attr_class] for attr_class in attributions]
        # average over samples 
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]

        # move the model also back to cpu (if not already on cpu)
        self.to('cpu')
        
        # combine into a single data frame 
        df_list = []
        layers = list(self.dataset.dataset.dat.keys())  # accessing multiomicdataset within tripletmultiomic dataset here
        for i in range(num_class):
            imp_layerwise = imp[i][0].split(layer_sizes, dim = 1)
            for j in range(len(layers)):
                features = self.dataset.dataset.features[layers[j]] # accessing multiomicdataset within tripletmultiomic dataset here
                importances = imp_layerwise[j][0].detach().numpy() # 0 => extract importances only for the anchor 
                df_list.append(pd.DataFrame({'target_variable': target_var, 'target_class': i, 'layer': layers[j], 'name': features, 'importance': importances}))    
        df_imp = pd.concat(df_list, ignore_index = True)
        
        # save scores in model
        self.feature_importances[target_var] = df_imp

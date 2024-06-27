# Supervised VAE-MMD architecture
import torch
import itertools 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np

import lightning as pl
from scipy import stats

from captum.attr import IntegratedGradients

from ..modules import *

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
    also can be attached to one ore more supervisor heads for regression/classification/survival tasks.
    In the absence of supervisor heads, it can be used for unsupervised learning. 

    Attributes:
        config (dict): Configuration settings for the model, including learning rates and dimensions.
        dataset: The MultiOmicDataset object containing the data and metadata.
        target_variables (list): A list of target variable names that the model aims to predict.
        batch_variables (list, optional): A list of variables used for batch correction. Defaults to None.
        surv_event_var (str, optional): The name of the survival event variable. Defaults to None.
        surv_time_var (str, optional): The name of the survival time variable. Defaults to None.
        use_loss_weighting (bool, optional): Whether to use loss weighting in the model. Defaults to True.
        device_type (str, optional): Type of device to run the model ('gpu' or 'cpu'). Defaults to None.
    """
    def __init__(self,  config, dataset, target_variables, batch_variables = None, 
                 surv_event_var = None, surv_time_var = None, use_loss_weighting = True,
                 device_type = None):
        super(supervised_vae, self).__init__()
        self.config = config
        self.dataset = dataset
        self.target_variables = target_variables
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        # both surv event and time variables are assumed to be numerical variables
        # we create only one survival variable for the pair (surv_time_var and surv_event_var)
        if self.surv_event_var is not None and self.surv_time_var is not None:
            self.target_variables = self.target_variables + [self.surv_event_var]
        self.batch_variables = batch_variables
        self.variables = self.target_variables + batch_variables if batch_variables else self.target_variables
        self.feature_importances = {}
        
        # sometimes the model may have exploding/vanishing gradients leading to NaN values
        self.nan_detected = False 

        self.device_type = device_type

        self.use_loss_weighting = use_loss_weighting
        
        if self.use_loss_weighting:
            # Initialize log variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict()
            for loss_type in itertools.chain(self.variables, ['mmd_loss']):
                self.log_vars[loss_type] = nn.Parameter(torch.zeros(1))
        
        layers = list(dataset.dat.keys())
        input_dims = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        # create a list of Encoder instances for separately encoding each omics layer
        self.encoders = nn.ModuleList([Encoder(input_dims[i], 
                                               # define hidden_dim size as a factor of input_dim  
                                               [int(input_dims[i] * config['hidden_dim_factor'])], 
                                               config['latent_dim']) for i in range(len(layers))])
        # Fully connected layers for concatenated means and log_vars
        self.FC_mean = nn.Linear(len(layers) * config['latent_dim'], config['latent_dim'])
        self.FC_log_var = nn.Linear(len(layers) * config['latent_dim'], config['latent_dim'])
        # list of decoders to decode each omics layer separately 
        self.decoders = nn.ModuleList([Decoder(config['latent_dim'], 
                                               [int(input_dims[i] * config['hidden_dim_factor'])], 
                                               input_dims[i]) for i in range(len(layers))])

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
        if self.dataset.variable_types[var] == 'numerical':
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
        Executes one training step using a single batch from the training dataset.

        Args:
            train_batch (tuple): The batch to train on, which includes input data and targets.
            batch_idx (int): Index of the current batch in the sequence.
            log (bool, optional): Whether to log the loss metrics to TensorBoard. Defaults to True.

        Returns:
            torch.Tensor: The total loss computed for the batch.
        """
        dat, y_dict = train_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        
        x_hat_list, z, mean, log_var, outputs = self.forward(x_list)
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        # compute loss values for the supervisor heads 
        losses = {'mmd_loss': mmd_loss}
        
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
        Executes one validation step using a single batch from the validation dataset.

        Args:
            val_batch (tuple): The batch to validate on, which includes input data and targets.
            batch_idx (int): Index of the current batch in the sequence.
            log (bool, optional): Whether to log the loss metrics to TensorBoard. Defaults to True.

        Returns:
            torch.Tensor: The total loss computed for the batch.
        """
        dat, y_dict = val_batch
        layers = dat.keys()
        x_list = [dat[x] for x in layers]
        
        x_hat_list, z, mean, log_var, outputs = self.forward(x_list)
        
        # compute mmd loss for each layer and take average
        mmd_loss_list = [self.MMD_loss(z.shape[1], z, x_hat_list[i], x_list[i]) for i in range(len(layers))]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        # compute loss values for the supervisor heads 
        losses = {'mmd_loss': mmd_loss}
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
        z.columns = [''.join(['E', str(x)]) for x in z.columns]
        z.index = dataset.samples
        return z
    
    def predict(self, dataset):
        """
        Evaluate the model on a dataset.

        Args:
            dataset (MultiOmicDataset): dataset containing input matrices for each omics layer.

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
        true_samples = torch.randn(200, latent_dim, device = self.device)
        mmd = self.compute_mmd(true_samples, z) # compute maximum mean discrepancy (MMD)
        nll = (xhat - x).pow(2).mean() #negative log likelihood
        return mmd+nll

    # Adaptor forward function for captum integrated gradients. 
    def forward_target(self, *args):
        input_data = list(args[:-2])  # one or more tensors (one per omics layer)
        target_var = args[-2]  # target variable of interest
        steps = args[-1]  # number of steps for IntegratedGradients().attribute 
        outputs_list = []
        for i in range(steps):
            # get list of tensors for each step into a list of tensors
            x_step = [input_data[j][i] for j in range(len(input_data))]
            x_hat_list, z, mean, log_var, outputs = self.forward(x_step)
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
        # Initialize the Integrated Gradients method
        ig = IntegratedGradients(self.forward_target)

        # Get the number of classes for the target variable
        if self.dataset.variable_types[target_var] == 'numerical':
            num_class = 1
        else:
            num_class = len(np.unique(self.dataset.ann[target_var]))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        aggregated_attributions = [[] for _ in range(num_class)]
        
        for batch in dataloader:
            dat, _ = batch
            x_list = [dat[x].to(device) for x in dat.keys()]
            input_data = tuple([data.unsqueeze(0).requires_grad_() for data in x_list])
            baseline = tuple(torch.zeros_like(x) for x in input_data)
            if num_class == 1:
                # returns a tuple of tensors (one per data modality)
                attributions = ig.attribute(input_data, baseline, 
                                             additional_forward_args=(target_var, steps), 
                                             n_steps=steps)
                aggregated_attributions[0].append(attributions)
            else:
                for target_class in range(num_class):
                    # returns a tuple of tensors (one per data modality)
                    attributions = ig.attribute(input_data, baseline, 
                                                 additional_forward_args=(target_var, steps), 
                                                 target=target_class, n_steps=steps)
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
        
        # summarize feature importances
        # Compute absolute attributions
        # Move the processed tensors to CPU for further operations that are not supported on GPU
        abs_attr = [[torch.abs(a).cpu() for a in attr_class] for attr_class in processed_attributions]
        # average over samples 
        imp = [[a.mean(dim=1) for a in attr_class] for attr_class in abs_attr]

        # move the model also back to cpu (if not already on cpu)
        self.to('cpu')

        # combine into a single data frame 
        df_list = []
        for i in range(num_class):
            for j in range(len(layers)):
                features = self.dataset.features[layers[j]]
                importances = imp[i][j][0].detach().numpy()
                target_class_label = dataset.label_mappings[target_var].get(i) if target_var in dataset.label_mappings else ''
                df_list.append(pd.DataFrame({'target_variable': target_var, 
                                             'target_class': i, 
                                             'target_class_label': target_class_label,
                                             'layer': layers[j], 
                                             'name': features, 'importance': importances}))  
        df_imp = pd.concat(df_list, ignore_index = True)
        
        # save scores in model
        self.feature_importances[target_var] = df_imp

    

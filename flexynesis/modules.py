# Networks that can be reused across different architectures

import torch
from torch import nn
from torch_geometric.nn import aggr, GCNConv, GATConv, SAGEConv, GraphConv, \
    global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gap


__all__ = ["Encoder", "Decoder", "MLP", "GNNs", "cox_ph_loss"]


class Encoder(nn.Module):
    """
    Encoder class for a Variational Autoencoder (VAE).
    
    The Encoder class is responsible for taking input data and generating the mean and
    log variance for the latent space representation.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        self.act = nn.LeakyReLU(0.2)
        
        hidden_layers = []
        
        hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        nn.init.xavier_uniform_(hidden_layers[-1].weight)
        hidden_layers.append(self.act)
        hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            nn.init.xavier_uniform_(hidden_layers[-1].weight)
            hidden_layers.append(self.act)
            hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))

        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.FC_mean  = nn.Linear(hidden_dims[-1], latent_dim)
        nn.init.xavier_uniform_(self.FC_mean.weight)
        self.FC_var   = nn.Linear(hidden_dims[-1], latent_dim)
        nn.init.xavier_uniform_(self.FC_var.weight)
        
    def forward(self, x):
        """
        Performs a forward pass through the Encoder network.
        
        Args:
            x (torch.Tensor): The input data tensor.
            
        Returns:
            mean (torch.Tensor): The mean of the latent space representation.
            log_var (torch.Tensor): The log variance of the latent space representation.
        """
        h_       = self.hidden_layers(x)
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        return mean, log_var
    
    
class Decoder(nn.Module):
    """
    Decoder class for a Variational Autoencoder (VAE).
    
    The Decoder class is responsible for taking the latent space representation and
    generating the reconstructed output data.
    """
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()

        self.act = nn.LeakyReLU(0.2)

        hidden_layers = []

        hidden_layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        nn.init.xavier_uniform_(hidden_layers[-1].weight)
        hidden_layers.append(self.act)
        hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            nn.init.xavier_uniform_(hidden_layers[-1].weight)
            hidden_layers.append(self.act)
            hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.FC_output = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.xavier_uniform_(self.FC_output.weight)

    def forward(self, x):
        """
        Performs a forward pass through the Decoder network.
        
        Args:
            x (torch.Tensor): The input tensor representing the latent space.
            
        Returns:
            x_hat (torch.Tensor): The reconstructed output tensor.
        """
        h = self.hidden_layers(x)
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for regression or classification tasks.
    
    The MLP class is a simple feed-forward neural network that can be used for regression
    when `output_dim` is set to 1 or for classification when `output_dim` is greater than 1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the MLP class with the given input dimension, output dimension, and hidden layer size.
        
        Args:
            input_dim (int): The input dimension.
            hidden_dim (int, optional): The size of the hidden layer. Default is 32.
            output_dim (int): The output dimension. Set to 1 for regression tasks, and > 1 for classification tasks.
        """
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, output_dim) if output_dim > 1 else nn.Linear(hidden_dim, 1, bias=False)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        Performs a forward pass through the MLP network.
        
        Args:
            x (torch.Tensor): The input data tensor.
            
        Returns:
            x (torch.Tensor): The output tensor after passing through the MLP network.
        """
        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the CNN model.

        This CNN has a single convolutional layer followed by batch normalization,
        ReLU activation, dropout, and another convolutional layer as output.

        Args:
            input_dim (int): The number of input dimensions or channels.
            hidden_dim (int): The number of hidden dimensions or channels after the first convolutional layer.
            output_dim (int): The number of output dimensions or channels after the output convolutional layer.
        """
        super().__init__()

        self.layer_1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """
        Define the forward pass of the CNN.

        The input tensor is passed through each layer of the network in sequence.
        The input tensor is first unsqueezed to add an extra dimension, then passed through
        the first convolutional layer, batch normalization, ReLU activation, dropout,
        and finally through the output convolutional layer before being squeezed back.

        Args:
            x (Tensor): A tensor of shape (N, C), where N is the batch size and C is the number of channels.
        Returns:
            Tensor: The output tensor of shape (N, C) after passing through the CNN.
        """
        # (N, C) -> (N, C, L) -> (N, C).
        x = x.unsqueeze(-1)
        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = x.squeeze(-1)
        return x
    
class GNNs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 conv='GC', act = 'relu'):
        super().__init__()
        
        act_options = {
            'relu': (torch.nn.ReLU()),
            'sigmoid': (torch.nn.Sigmoid()),
            'tanh': (torch.nn.Tanh()),
            'gelu': (torch.nn.GELU())
        }
        # check if the activation function string is valid
        if act not in act_options:
            raise ValueError("Invalid activation function string. Choose from 'relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu', 'elu', or 'gelu'.")
        
        # instantiate the activation function
        self.activation = act_options[act]
        self.convs = torch.nn.ModuleList()
        conv_options = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
            'GC': GraphConv
        }
        if conv not in conv_options:
            raise ValueError('Unknown convolution type. Choose one of: ',conv_options.keys())
        
        self.conv = conv_options[conv]
        self.layer_1 = self.conv(input_dim, hidden_dim)
        self.act_1 = self.activation
        self.layer_2 = self.conv(hidden_dim, output_dim)
        self.act_2 = self.activation
        self.aggregation = aggr.SumAggregation()

    def forward(self, x, edge_index, batch):
        x = self.layer_1(x, edge_index)
        x = self.act_1(x)
        x = self.layer_2(x, edge_index)
        x = self.act_2(x)
        x = self.aggregation(x, batch)
        return x
    
def cox_ph_loss(outputs, durations, events):
    """
    Calculate the Cox proportional hazards loss.

    Args:
        outputs (torch.Tensor): The output log-risk scores from the MLP.
        durations (torch.Tensor): The observed times (durations) for each sample.
        events (torch.Tensor): The event indicators (1 if event occurred, 0 if censored) for each sample.

    Returns:
        torch.Tensor: The calculated CoxPH loss.
    """
    valid_indices = ~torch.isnan(durations) & ~torch.isnan(events)
    if valid_indices.sum() > 0:
        outputs = outputs[valid_indices]
        events = events[valid_indices]
        durations = durations[valid_indices]
        
        # Exponentiate the outputs to get the hazard ratios
        hazards = torch.exp(outputs)
        # Ensure hazards is at least 1D
        if hazards.dim() == 0:
            hazards = hazards.unsqueeze(0)  # Make hazards 1D if it's a scalar
        # Calculate the risk set sum
        log_risk_set_sum = torch.log(torch.cumsum(hazards[torch.argsort(durations, descending=True)], dim=0))
        # Get the indices that sort the durations in descending order
        sorted_indices = torch.argsort(durations, descending=True)
        events_sorted = events[sorted_indices]

        # Calculate the loss
        uncensored_loss = torch.sum(outputs[sorted_indices][events_sorted == 1]) - torch.sum(log_risk_set_sum[events_sorted == 1])
        total_loss = -uncensored_loss / torch.sum(events)
    else: 
        total_loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
    if not torch.isfinite(total_loss):
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    return total_loss
# Networks that can be reused across different architectures

import torch
from torch import nn
from torch_geometric.nn import aggr, GCNConv, GATConv, SAGEConv, GraphConv


__all__ = ["Encoder", "Decoder", "MLP", "flexGCN", "cox_ph_loss"]


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
    
class flexGCN(nn.Module):
    """
    A Graph Neural Network (GNN) model using configurable convolution and activation layers.

    This class defines a GNN that can utilize various graph convolution types and activation functions.
    It supports a configurable number of convolutional layers with batch normalization and dropout
    for regularization. The model aggregates node features into a single vector per graph using
    a fully connected layer.

    Attributes:
        act (torch.nn.Module): Activation function applied after each convolution.
        convs (nn.ModuleList): List of convolutional layers.
        bns (nn.ModuleList): List of batch normalization layers applied after each convolution.
        dropout (nn.Dropout): Dropout layer applied after activation to prevent overfitting.
        fc (torch.nn.Linear): Fully connected layer that aggregates node features into a single vector.

    Args:
        node_count (int): The number of nodes in each graph.
        node_feature_count (int): The number of features each node initially has.
        node_embedding_dim (int): The size of the node embeddings (output dimension of the convolutions).
        output_dim (int): The size of the output vector, which is the final feature vector for the whole graph.
        num_convs (int, optional): Number of convolutional layers in the network. Defaults to 2.
        dropout_rate (float, optional): The dropout probability used for regularization. Defaults to 0.2.
        conv (str, optional): Type of convolution layer to use. Supported types include 'GCN' for Graph Convolution Network, 
                              'GAT' for Graph Attention Network, 'SAGE' for GraphSAGE, and 'GC' for generic Graph Convolution. 
                              Defaults to 'GC'.
        act (str, optional): Type of activation function to use. Supported types include 'relu', 'sigmoid', 
                             'leakyrelu', 'tanh', and 'gelu'. Defaults to 'relu'.

    Raises:
        ValueError: If an unsupported activation function or convolution type is specified.

    Example:
        >>> model = flexGCN(node_count=100, node_feature_count=5, node_embedding_dim=64, output_dim=10, 
                         num_convs=3, dropout_rate=0.3, conv='GAT', act='relu')
        >>> output = model(input_features, edge_index)
        # Where `input_features` is a tensor of shape (batch_size, num_nodes, node_feature_count)
        # and `edge_index` is a list of edges in the COO format (2, num_edges).
    """
    def __init__(self, node_count, node_feature_count, node_embedding_dim, output_dim, 
                 num_convs = 2, dropout_rate = 0.2, conv='GC', act='relu'):
        super().__init__()

        act_options = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(), 
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        if act not in act_options:
            raise ValueError("Invalid activation function string. Choose from ", list(act_options.keys()))
        
        conv_options = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
            'GC': GraphConv
        }
        if conv not in conv_options:
            raise ValueError('Unknown convolution type. Choose one of: ', list(conv_options.keys()))

        self.act = act_options[act]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize the first convolution layer separately if different input size
        self.convs.append(conv_options[conv](node_feature_count, node_embedding_dim))
        self.bns.append(nn.BatchNorm1d(node_embedding_dim))

        # Loop to create the remaining convolution and BN layers
        for _ in range(1, num_convs):
            self.convs.append(conv_options[conv](node_embedding_dim, node_embedding_dim))
            self.bns.append(nn.BatchNorm1d(node_embedding_dim))

        # Final fully connected layer
        self.fc = nn.Linear(node_embedding_dim * node_count, output_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x.view(-1, x.size(2))).view_as(x)
            x = self.act(x)
            x = self.dropout(x) 

        # Flatten the output of all nodes into a single vector per graph/sample
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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

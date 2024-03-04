# Networks that can be reused across different architectures

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, GATConv, GINConv, PNAConv, SAGEConv, ChebConv, \
    global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gap


__all__ = ["Encoder", "Decoder", "MLP", "EmbeddingNetwork", "GCNN", "cox_ph_loss"]


class Encoder(nn.Module):
    """
    Encoder class for a Variational Autoencoder (VAE).
    
    The Encoder class is responsible for taking input data and generating the mean and
    log variance for the latent space representation.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        hidden_layers = []
        
        hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        nn.init.xavier_uniform_(hidden_layers[-1].weight)
        hidden_layers.append(self.LeakyReLU)
        hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            nn.init.xavier_uniform_(hidden_layers[-1].weight)
            hidden_layers.append(self.LeakyReLU)
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

        self.LeakyReLU = nn.LeakyReLU(0.2)

        hidden_layers = []

        hidden_layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        nn.init.xavier_uniform_(hidden_layers[-1].weight)
        hidden_layers.append(self.LeakyReLU)
        hidden_layers.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            nn.init.xavier_uniform_(hidden_layers[-1].weight)
            hidden_layers.append(self.LeakyReLU)
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


class EmbeddingNetwork(nn.Module):
    """
    A simple feed-forward neural network for generating embeddings.
    
    The EmbeddingNetwork class is a straightforward feed-forward network
    that can be used to generate embeddings from input data.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the EmbeddingNetwork class with the given input size, hidden layer size, and output size.
        
        Args:
            input_size (int): The size of the input data.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer, representing the dimensionality of the embeddings.
        """
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs a forward pass through the EmbeddingNetwork.
        
        Args:
            x (torch.Tensor): The input data tensor.
            
        Returns:
            x (torch.Tensor): The output tensor representing the generated embeddings.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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

class GraphNNs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv='CHEB',
                 dropout=0.5, number_layers=5, device=None, deg=None, act = None):
        super().__init__()
        """
        Initialize the GNNs model.

        This model consists of two Graph Convolutional layers, each followed by a ReLU activation.
        After the second convolutional layer, the features of nodes are aggregated.

        Args:
            input_dim (int): The number of input dimensions or features.
            hidden_dim (int): The number of hidden dimensions or features after the first graph convolutional layer.
            output_dim (int): The number of output dimensions or features after the second graph convolutional layer.
        """
        self.device = device

        act_options = {
            'relu': (torch.nn.ReLU()),
            'sigmoid': (torch.nn.Sigmoid()),
            'tanh': (torch.nn.Tanh()),
            'softmax': (torch.nn.Softmax(dim=None)),
            'leakyrelu': (torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)),
            'elu': (torch.nn.ELU(alpha=1.0, inplace=False)),
            'gelu': (torch.nn.GELU())
        }
        # check if the activation function string is valid
        if act not in act_options:
            raise ValueError("Invalid activation function string. Choose from 'relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu', 'elu', or 'gelu'.")
        
        # instantiate the activation function
        self.activation = act_options[act]

        self.dropout = torch.nn.Dropout(dropout)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        conv_options = {
            'GCN': (GCNConv(input_dim, input_dim)),
            'GAT': (GATConv(input_dim, input_dim)),
            'GIN': (GINConv(nn.Sequential(nn.Linear(input_dim, input_dim).to(self.device), 
                                                   torch.nn.BatchNorm1d(input_dim).to(self.device), 
                                                   nn.ReLU(), nn.Linear(input_dim, input_dim).to(self.device)))),
            #'PNA': (PNAConv(input_dim, input_dim, aggregators=['mean', 'min', 'max'], 
            #                         scalers=['identity', 'amplification', 'attenuation'], deg=deg, towers=1, 
            #                         pre_layers=1, post_layers=1)),
            'SAGE': (SAGEConv(input_dim, input_dim)),
            'CHEB': (ChebConv(input_dim, input_dim, K=2)),
            #'MMA': (MMAConv(num_encoded_features+44, num_encoded_features+44, aggregators=['mean', 'min', 'max'], 
            #                         scalers=['identity', 'amplification', 'attenuation'], deg=deg, towers=1, 
            #                         pre_layers=1, post_layers=1, mask = True, device = self.device)),
            #'GMN': (GMNConv(num_encoded_features+44, num_encoded_features+44, aggregators=['mean', 'min', 'max'], 
            #                         scalers=['identity', 'amplification', 'attenuation'], deg=deg, towers=1, 
            #                         pre_layers=1, post_layers=1))
        }
        if conv not in conv_options:
            raise ValueError('Unknown convolution type')
        
        self.conv = conv_options[conv].to(self.device)
        
        for i in range(number_layers):
            self.convs.append(self.conv)
            self.bns.append(nn.BatchNorm1d(input_dim).to(self.device))

        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.bn_ff1 = nn.BatchNorm1d(hidden_dim).to(self.device)

        self.fc2 = nn.Linear(hidden_dim, output_dim).to(self.device)


    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def extract_embeddings(self, x, edge_index, batch):
        # get graph input
        #batch, edge_attr, inputs, output, random_walk_pe = data.batch, data.edge_attr, data.inputs, data.output, data.random_walk_pe

        #x = self.atom_encoder(x.int())
        #x = torch.cat([x, random_walk_pe], dim=-1)

        for conv, batch_norm in zip(self.convs, self.bns):
            if conv == "GCN" or conv == "CHEB" or conv == "GAT":
                x = self.dropout(self.activation(batch_norm(conv(x, edge_index))))
            else:
                x = self.dropout(self.activation(batch_norm(conv(x, edge_index))))
        
        return x
    
    def get_attention_scores(self, x, edge_index, batch):

        #batch, edge_attr, inputs, output, random_walk_pe = data.batch, data.edge_attr, data.inputs, data.output, data.random_walk_pe

        attention_scores_list = []

        #x = self.atom_encoder(x.int())
        #x = torch.cat([x, random_walk_pe], dim=-1)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.bns)):

            x, attention_scores = conv(x, edge_index, return_attention_weights=True)  # adapt as per your GAT layer's API
            attention_scores_list.append((i, attention_scores))

        return attention_scores_list

    def forward(self, x, edge_index, batch):
        """
        Define the forward pass of the GraphNNs.

        The input graph data is processed through two graph convolutional layers with ReLU activation.
        Finally, the node features are aggregated.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, input_dim].
            edge_index (LongTensor): The edge indices in COO format with shape [2, num_edges].
            batch (LongTensor): The batch vector which assigns each node to a specific example in the batch.

        Returns:
            Tensor: The output tensor after processing through the GCNN, with shape [num_nodes, output_dim].
        """
        for conv, batch_norm in zip(self.convs, self.bns):
            if conv == "GCN" or conv == "CHEB" or conv == "GAT":
                x = self.dropout(self.activation(batch_norm(conv(x, edge_index))))
            else:
                x = self.dropout(self.activation(batch_norm(conv(x, edge_index))))
        x = gap(x, batch)

        x = self.activation(self.bn_ff1(self.fc1(x)))
        x = self.dropout(x)

        out = self.fc2(x)

        return out



class GCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        """
        Initialize the GCNN model.

        This model consists of two Graph Convolutional layers, each followed by a ReLU activation.
        After the second convolutional layer, the features of nodes are aggregated.

        Args:
            input_dim (int): The number of input dimensions or features.
            hidden_dim (int): The number of hidden dimensions or features after the first graph convolutional layer.
            output_dim (int): The number of output dimensions or features after the second graph convolutional layer.
        """
        self.layer_1 = gnn.GraphConv(input_dim, hidden_dim)
        self.relu_1 = nn.ReLU()
        self.layer_2 = gnn.GraphConv(hidden_dim, output_dim)
        self.relu_2 = nn.ReLU()
        self.aggregation = gnn.aggr.SumAggregation()

    def forward(self, x, edge_index, batch):
        """
        Define the forward pass of the GCNN.

        The input graph data is processed through two graph convolutional layers with ReLU activation.
        Finally, the node features are aggregated.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, input_dim].
            edge_index (LongTensor): The edge indices in COO format with shape [2, num_edges].
            batch (LongTensor): The batch vector which assigns each node to a specific example in the batch.

        Returns:
            Tensor: The output tensor after processing through the GCNN, with shape [num_nodes, output_dim].
        """
        #print('batch:', batch.x)
        x = self.layer_1(x, edge_index)
        x = self.relu_1(x)
        x = self.layer_2(x, edge_index)
        x = self.relu_2(x)
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
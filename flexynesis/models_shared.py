# Networks that can be reused across different architectures

import torch
from torch import nn


class Encoder(nn.Module):
    """
    Encoder class for a Variational Autoencoder (VAE).
    
    The Encoder class is responsible for taking input data and generating the mean and
    log variance for the latent space representation.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the Encoder class with given input dimensions, hidden layer dimensions, and latent dimensions.
        
        Args:
            input_dim (int): The dimensionality of the input data.
            hidden_dims (list): A list of integers representing the dimensions of the hidden layers.
            latent_dim (int): The dimensionality of the latent space.
        """
        super(Encoder, self).__init__()

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # Create a list to store the hidden layers
        hidden_layers = []
        
        # Add the input layer to the first hidden layer
        hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        hidden_layers.append(self.LeakyReLU)

        # Create the hidden layers
        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(self.LeakyReLU)

        # Add the hidden layers to the model using nn.Sequential
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.FC_mean  = nn.Linear(hidden_dims[-1], latent_dim)
        self.FC_var   = nn.Linear(hidden_dims[-1], latent_dim)
        
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
        """
        Initializes the Decoder class with given latent dimensions, hidden layer dimensions, and output dimensions.
        
        Args:
            latent_dim (int): The dimensionality of the latent space.
            hidden_dims (list): A list of integers representing the dimensions of the hidden layers.
            output_dim (int): The dimensionality of the output data.
        """
        super(Decoder, self).__init__()

        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Create a list to store the hidden layers
        hidden_layers = []

        # Add the input layer to the first hidden layer
        hidden_layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        hidden_layers.append(self.LeakyReLU)

        # Create the hidden layers
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(self.LeakyReLU)

        # Add the hidden layers to the model using nn.Sequential
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.FC_output = nn.Linear(hidden_dims[-1], output_dim)

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
        self.layer_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.4)
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
        x = self.dropout(x)
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
    
# Simple feed-forward multi-class classifier
class Classifier(nn.Module):
    """
    A simple feed-forward neural network for multi-class classification tasks.
    
    The Classifier class is a straightforward feed-forward network that can be used
    to perform multi-class classification on input data.
    """
    def __init__(self, input_size, hidden_dims, num_classes):
        """
        Initializes the Classifier class with the given input size, hidden layer dimensions, and number of classes.
        
        Args:
            input_size (int): The size of the input data.
            hidden_dims (list): A list of integers representing the dimensions of the hidden layers.
            num_classes (int): The number of output classes.
        """
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))

    def forward(self, x):
        """
        Performs a forward pass through the Classifier network.
        
        Args:
            x (torch.Tensor): The input data tensor.
            
        Returns:
            x (torch.Tensor): The output tensor after passing through the Classifier network.
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
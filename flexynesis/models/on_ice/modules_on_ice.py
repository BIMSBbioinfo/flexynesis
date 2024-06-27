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

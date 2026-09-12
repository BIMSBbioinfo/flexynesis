# Networks that can be reused across different architectures

import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv

__all__ = ["Encoder", "Decoder", "MLP", "flexGCN", "GraphMAEHead", "cox_ph_loss"]

CONV_OPTIONS = {
    "GCN": GCNConv,
    "GAT": GATConv,
    "SAGE": SAGEConv,
    "GC": GraphConv,
}
# SAGEConv aggregates over neighbours with no slot for a scalar weight, and
# GATConv learns its own attention, so only these two can take an edge weight.
EDGE_WEIGHTED_CONVS = ("GCN", "GC")


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

        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            nn.init.xavier_uniform_(hidden_layers[-1].weight)
            hidden_layers.append(self.act)
            hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.FC_mean = nn.Linear(hidden_dims[-1], latent_dim)
        nn.init.xavier_uniform_(self.FC_mean.weight)
        self.FC_var = nn.Linear(hidden_dims[-1], latent_dim)
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
        h_ = self.hidden_layers(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
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
            hidden_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

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
        super().__init__()
        hidden_dim = max(hidden_dim, 2)  # make sure there are at least 2 units
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_out = (
            nn.Linear(hidden_dim, output_dim)
            if output_dim > 1
            else nn.Linear(hidden_dim, 1, bias=False)
        )
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
        conv (str, optional): Type of convolution layer to use. Supported types
                              include 'GCN' for Graph Convolution Network,
                              'SAGE' for GraphSAGE, and 'GC' for generic Graph Convolution.
                              Defaults to 'GC'.
        act (str, optional): Type of activation function to use. Supported types
                             include 'relu', 'sigmoid', 'leakyrelu', 'tanh',
                             and 'gelu'. Defaults to 'relu'.

    Raises:
        ValueError: If an unsupported activation function or convolution type is specified.

    Example:
        >>> model = flexGCN(node_count=100, node_feature_count=5, node_embedding_dim=64, output_dim=10,
                         num_convs=3, dropout_rate=0.3, conv='SAGE', act='relu')
        >>> output = model(input_features, edge_index)
        # Where `input_features` is a tensor of shape (batch_size, num_nodes, node_feature_count)
        # and `edge_index` is a list of edges in the COO format (2, num_edges).
    """

    def __init__(
        self,
        node_count,
        node_feature_count,
        node_embedding_dim,
        output_dim,
        num_convs=2,
        dropout_rate=0.2,
        conv="GC",
        act="relu",
        readout="dim_attention",
        add_self_loops=True,
    ):
        super().__init__()

        act_options = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        if act not in act_options:
            raise ValueError(
                "Invalid activation function string. Choose from ",
                list(act_options.keys()),
            )

        conv_options = CONV_OPTIONS
        if conv not in conv_options:
            raise ValueError(
                "Unknown convolution type. Choose one of: ",
                list(conv_options.keys()),
            )

        self.act = act_options[act]
        self.supports_edge_weight = conv in EDGE_WEIGHTED_CONVS

        # GCN's renormalisation trick is defined on A + I (Kipf & Welling
        # 2017), so self-loops are on by default. With them off, a node's
        # representation is built purely from its neighbours and an isolated
        # node contributes nothing. GraphConv keeps a separate root weight and
        # SAGEConv concatenates the node's own features, so neither accepts the
        # argument and for those two a self-connection is always present.
        conv_kwargs = ({"add_self_loops": add_self_loops}
                       if conv in ("GCN", "GAT") else {})
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize the first convolution layer separately if different input size
        self.convs.append(conv_options[conv](node_feature_count,
                                             node_embedding_dim, **conv_kwargs))
        self.bns.append(nn.BatchNorm1d(node_embedding_dim))

        # Loop to create the remaining convolution and BN layers
        for _ in range(1, num_convs):
            self.convs.append(
                conv_options[conv](node_embedding_dim, node_embedding_dim,
                                   **conv_kwargs)
            )
            self.bns.append(nn.BatchNorm1d(node_embedding_dim))

        readout_options = ("mean", "sum", "max", "meanmax", "attention",
                           "dim_attention", "flatten")
        if readout not in readout_options:
            raise ValueError(
                "Unknown readout. Choose one of: ", list(readout_options)
            )
        self.readout = readout

        # Width of the sample representation the final layer sees. The pooling
        # readouts collapse the node axis entirely; "dim_attention" leaves one
        # value per node and "flatten" leaves a node's full embedding.
        if readout == "meanmax":
            fc_in = node_embedding_dim * 2
        elif readout == "dim_attention":
            fc_in = node_count
        elif readout == "flatten":
            fc_in = node_count * node_embedding_dim
        else:
            fc_in = node_embedding_dim

        # Attention pooling scores each node from its own embedding, never
        # from its index, so pooling stays permutation-equivariant: relabel the
        # nodes and the sample vector is unchanged. Unlike mean pooling it can
        # still weight informative nodes above uninformative ones.
        self.attention = (
            nn.Linear(node_embedding_dim, 1) if readout == "attention" else None
        )

        # "dim_attention" attends over the embedding *dimensions* of each
        # node rather than over the nodes, collapsing each node to one number
        # and concatenating those. Node identity survives as position, which
        # node pooling discards, at 1/node_embedding_dim of the width of
        # "flatten". Scores come from the node's own embedding through a layer
        # shared across nodes, so which dimension wins can differ per node
        # without a per-node parameter.
        self.dim_attention = (
            nn.Linear(node_embedding_dim, node_embedding_dim)
            if readout == "dim_attention" else None
        )
        self.last_attention = None

        # The readout is projected into a latent space of size output_dim,
        # which is what the downstream heads consume.
        self.fc = nn.Linear(fc_in, output_dim)
        self.output_dim = output_dim

    def encode_nodes(self, x, edge_index, edge_weight=None):
        """Message passing only: returns one embedding per node, unpooled.

        Kept separate from the readout so auxiliary heads that work per node
        (e.g. GraphMAEHead) can reuse the same encoder pass.

        Args:
            x (torch.Tensor): Node features of shape (batch, node_count, node_feature_count).
            edge_index (torch.Tensor): Graph connectivity in COO format.
            edge_weight (torch.Tensor, optional): Edge weights; only used by
                convolution types that support them (a warning is issued and the
                weights are ignored otherwise).

        Returns:
            torch.Tensor: Per-node embeddings of shape
                (batch, node_count, node_embedding_dim).
        """
        if edge_weight is not None and not self.supports_edge_weight:
            warnings.warn(
                "Edge weights were provided but this convolution type ignores "
                "them; use GCN or GC to make use of them."
            )
            edge_weight = None
        for conv, bn in zip(self.convs, self.bns):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            x = bn(x.view(-1, x.size(2))).view_as(x)
            x = self.act(x)
            x = self.dropout(x)

        return x

    def pool(self, x):
        """Reduce the per-node embeddings to one vector per graph/sample.

        The reduction follows ``self.readout`` (mean/sum/max/attention/
        dim_attention/flatten/meanmax) and the pooled vector is projected
        through ``self.fc`` into the latent space of size ``output_dim``.

        Args:
            x (torch.Tensor): Per-node embeddings of shape
                (batch, node_count, node_embedding_dim).

        Returns:
            torch.Tensor: Pooled embedding of shape (batch, output_dim).
        """
        if self.readout == "mean":
            x = x.mean(dim=1)
        elif self.readout == "sum":
            x = x.sum(dim=1)
        elif self.readout == "max":
            x = x.max(dim=1).values
        elif self.readout == "attention":
            weights = torch.softmax(self.attention(x), dim=1)
            self.last_attention = weights.detach()
            x = (weights * x).sum(dim=1)
        elif self.readout == "dim_attention":
            # softmax over dim=2: the embedding dimensions of each node
            weights = torch.softmax(self.dim_attention(x), dim=2)
            self.last_attention = weights.detach()
            x = (weights * x).sum(dim=2)
        elif self.readout == "flatten":
            # Concatenate every node's full embedding, so the sample vector
            # is node_count * node_embedding_dim wide and node i occupies a
            # fixed slice. Customary when one graph is shared across samples
            # and each sample is a signal over its nodes (Defferrard et al.
            # 2016; Chereda et al. 2019). Unlike the pooling readouts it ties
            # weights to node indices, which keeps node identity but costs
            # permutation equivariance and scales with the node count.
            x = x.reshape(x.size(0), -1)
        else:  # meanmax
            x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        x = self.fc(x)
        return x

    def forward(self, x, edge_index, edge_weight=None, return_nodes=False):
        nodes = self.encode_nodes(x, edge_index, edge_weight)
        pooled = self.pool(nodes)
        return (pooled, nodes) if return_nodes else pooled


class GraphMAEHead(nn.Module):
    """Masked node-feature reconstruction (GraphMAE, Hou et al. 2022, KDD).

    A random subset of each sample's nodes has its input value replaced by a
    learned mask token; the encoder runs as usual; the masked nodes' embeddings
    are then zeroed again ("re-masking") and a one-layer graph convolution has
    to rebuild the hidden values from the neighbours alone. Used as an
    auxiliary objective alongside a supervised head, it pushes the encoder to
    propagate information between neighbouring nodes.

    The reconstruction target is the node features, not the adjacency matrix as
    in the Kipf & Welling graph auto-encoder. Adjacency reconstruction is
    invariant to a relabelling of the nodes, so it cannot express whether the
    identities attached to the nodes are the right ones.
    """

    def __init__(
        self,
        node_feature_count,
        node_embedding_dim,
        mask_ratio=0.5,
        loss="auto",
        gamma=2.0,
        conv="GCN",
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if conv not in CONV_OPTIONS:
            raise ValueError("Unknown convolution type. Choose one of: ",
                             list(CONV_OPTIONS))
        self.mask_ratio = mask_ratio
        self.gamma = gamma

        # GraphMAE scores reconstruction with the scaled cosine error, which
        # compares the *direction* of a node's feature vector. The cosine of
        # two scalars is always +-1, so that is degenerate when a node carries
        # a single feature (one omics layer); fall back to MSE there.
        if loss == "auto":
            loss = "sce" if node_feature_count > 1 else "mse"
        if loss not in ("sce", "mse"):
            raise ValueError("Unknown reconstruction loss. Choose 'sce', "
                             "'mse' or 'auto'")
        self.loss_type = loss

        self.mask_token = nn.Parameter(torch.zeros(node_feature_count))
        self.enc_to_dec = nn.Linear(node_embedding_dim, node_embedding_dim,
                                    bias=False)
        self.decoder = CONV_OPTIONS[conv](node_embedding_dim,
                                          node_feature_count)
        self.supports_edge_weight = conv in EDGE_WEIGHTED_CONVS

    def apply_mask(self, x, generator=None):
        """Replace a random subset of each sample's nodes with the mask token.

        The subset is drawn per sample, so a node is hidden in some samples and
        visible in others and no node is permanently unsupervised.

        Args:
            x (torch.Tensor): Node features of shape
                (batch, node_count, node_feature_count).
            generator (torch.Generator, optional): Random source for
                reproducible masking.

        Returns:
            tuple: ``(masked_x, mask)`` where ``masked_x`` has the same shape as
                ``x`` with ``k = max(1, round(mask_ratio * node_count))`` nodes per
                sample replaced by ``self.mask_token``, and ``mask`` is a boolean
                tensor of shape (batch, node_count) marking the hidden nodes.
        """
        batch, node_count, _ = x.shape
        k = max(1, int(round(self.mask_ratio * node_count)))
        scores = torch.rand(batch, node_count, device=x.device,
                            generator=generator)
        idx = scores.argsort(dim=1)[:, :k]
        mask = torch.zeros(batch, node_count, dtype=torch.bool,
                           device=x.device)
        mask.scatter_(1, idx, True)
        token = self.mask_token.to(x.dtype).view(1, 1, -1)
        return torch.where(mask.unsqueeze(-1), token, x), mask

    def reconstruct(self, nodes, edge_index, mask, edge_weight=None):
        nodes = self.enc_to_dec(nodes)
        # Re-masking: a masked node's own embedding is zeroed before decoding,
        # so its value can only be rebuilt by travelling along edges.
        nodes = nodes.masked_fill(mask.unsqueeze(-1), 0.0)
        if edge_weight is None or not self.supports_edge_weight:
            return self.decoder(nodes, edge_index)
        return self.decoder(nodes, edge_index, edge_weight)

    def loss(self, x, x_hat, mask):
        target, pred = x[mask], x_hat[mask]
        if target.numel() == 0:
            return torch.zeros((), device=x.device)
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        cos = F.cosine_similarity(pred, target, dim=-1)
        return (1.0 - cos).pow(self.gamma).mean()


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
        log_risk_set_sum = torch.log(
            torch.cumsum(hazards[torch.argsort(durations, descending=True)], dim=0)
        )
        # Get the indices that sort the durations in descending order
        sorted_indices = torch.argsort(durations, descending=True)
        events_sorted = events[sorted_indices]

        # Calculate the loss
        uncensored_loss = torch.sum(
            outputs[sorted_indices][events_sorted == 1]
        ) - torch.sum(log_risk_set_sum[events_sorted == 1])
        total_loss = -uncensored_loss / torch.sum(events)
    else:
        total_loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
    if not torch.isfinite(total_loss):
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    return total_loss

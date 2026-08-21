# config.py
from skopt.space import Categorical, Integer, Real

epochs = [500]

search_spaces = {
    "DirectPred": [
        Integer(16, 128, name="latent_dim"),
        Real(
            0.2, 0.5, name="hidden_dim_factor"
        ),  # relative size of the hidden_dim w.r.t input_dim
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Integer(8, 32, name="supervisor_hidden_dim"),
        Categorical(epochs, name="epochs"),
    ],
    "supervised_vae": [
        Integer(16, 128, name="latent_dim"),
        Real(
            0.2, 0.5, name="hidden_dim_factor"
        ),  # relative size of the hidden_dim w.r.t input_dim
        Integer(8, 32, name="supervisor_hidden_dim"),
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Categorical(epochs, name="epochs"),
    ],
    "CrossModalPred": [
        Integer(16, 128, name="latent_dim"),
        Real(
            0.2, 0.5, name="hidden_dim_factor"
        ),  # relative size of the hidden_dim w.r.t input_dim
        Integer(8, 32, name="supervisor_hidden_dim"),
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Categorical(epochs, name="epochs"),
    ],
    "MultiTripletNetwork": [
        Integer(16, 128, name="latent_dim"),
        Real(
            0.2, 0.5, name="hidden_dim_factor"
        ),  # relative size of the hidden_dim w.r.t input_dim
        Integer(8, 32, name="supervisor_hidden_dim"),
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Categorical(epochs, name="epochs"),
    ],
    "GNN": [
        Integer(16, 128, name="latent_dim"),
        Integer(4, 32, name="node_embedding_dim"),  # node embedding dimensions
        Integer(1, 4, name="num_convs"),  # number of convolutional layers
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Integer(8, 32, name="supervisor_hidden_dim"),
        Categorical(epochs, name="epochs"),
        Categorical(["relu"], name="activation"),
    ],
    "DeepTSP": [
        Integer(8, 64, name="hidden_dim"),  # SetEncoder hidden layer width
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Categorical(epochs, name="epochs"),
        Integer(5, 50, name="target_k"),  # total surviving pairs, tied to assay budget
        Integer(2, 10, name="prune_every"),  # epochs between hard-pruning rounds
        Integer(10, 100, name="max_genes_per_set"),  # per-set Laplacian cap
    ],
}

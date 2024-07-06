# config.py
from skopt.space import Integer, Categorical, Real

epochs = [500]

search_spaces = {
    'DirectPred': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 1, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Categorical(epochs, name='epochs')
    ], 
    'supervised_vae': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 1, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'CrossModalPred': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 1, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'MultiTripletNetwork': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 1, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'GNN': [
        Integer(16, 128, name='latent_dim'),
        Integer(4, 32, name='node_embedding_dim'), # node embedding dimensions
        Integer(1, 4, name='num_convs'), # number of convolutional layers 
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Categorical(epochs, name='epochs'),
        Categorical(['relu'], name="activation")
    ]
}

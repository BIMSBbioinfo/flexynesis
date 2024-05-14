# config.py
from skopt.space import Integer, Categorical, Real

epochs = [200]

search_spaces = {
    'DirectPred': [
        Integer(16, 128, name='latent_dim'),
        Real(0.5, 2, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(16, 128, name='supervisor_hidden_dim'),
        Categorical(epochs, name='epochs')
    ], 
    'supervised_vae': [
        Integer(16, 128, name='latent_dim'),
        Real(0.5, 2, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(16, 128, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'CrossModalPred': [
        Integer(16, 128, name='latent_dim'),
        Real(0.5, 2, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'MultiTripletNetwork': [
        Integer(16, 128, name='latent_dim'),
        Real(0.5, 2, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    "DirectPredGCNN": [
        Integer(16, 128, name="latent_dim"),
        Real(0.5, 2, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Categorical(epochs, name="epochs"),
        Integer(16, 128, name='supervisor_hidden_dim'),
        Categorical(['relu'], name="activation")
    ]
}

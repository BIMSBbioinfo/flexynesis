# config.py
from skopt.space import Integer, Categorical, Real

search_spaces = {
    'DirectPred': [
        Integer(16, 128, name='latent_dim'),
        Integer(64, 512, name='hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical([100, 200], name='epochs')
    ], 
    'SVAE': [
        Integer(16, 128, name='latent_dim'),
        Integer(64, 512, name='hidden_dim'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical([100, 200], name='epochs')
    ],
    'MultiTripletNetwork': [
        Integer(16, 128, name='embedding_dim'),
        Integer(64, 512, name='hidden_dim'),
        Integer(8, 32, name='classifier_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical([100, 200], name='epochs')
    ]
}

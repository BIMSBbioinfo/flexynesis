# config.py
from skopt.space import Integer, Categorical, Real

space_model1 = [
    Categorical([16, 32, 48], name='latent_dim'),
    Categorical([32, 64, 128], name='hidden_dim'),
    Real(0.0001, 0.01, prior='log-uniform', name='lr'),
    Categorical([32, 64, 128], name='batch_size'),
    Categorical([100, 150, 200], name='epochs')
]

space_model2 = [
    Categorical([16, 32, 48], name='latent_dim'),
    Categorical([32, 64, 128], name='hidden_dim'),
    Categorical([4, 8, 16], name='supervisor_hidden_dim'),
    Real(0.0001, 0.01, prior='log-uniform', name='lr'),
    Categorical([32, 64, 128], name='batch_size'),
    Categorical([100, 150, 200], name='epochs')
]

space_model3 = [
    Categorical([16, 48, 96], name='embedding_dim'),
    Categorical([32, 128, 512], name='hidden_dim'),
    Categorical([4, 16], name='classifier_hidden_dim'),
    Real(0.0001, 0.01, prior='log-uniform', name='lr'),
    Categorical([64, 128], name='batch_size'),
    Categorical([100, 200, 300, 400, 500], name='epochs')
]

search_spaces = {
    'DirectPred': space_model1,
    'SVAE': space_model2,
    'MultiTripletNetwork': space_model3
}

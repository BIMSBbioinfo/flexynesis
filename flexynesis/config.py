# config.py
from skopt.space import Integer, Categorical

space_model1 = [
    Integer(16, 48, name='latent_dim'),
    Integer(32, 128, name='hidden_dim'),
    Categorical([0.01, 0.001, 0.0001], name='lr'),
    Integer(32, 128, name='batch_size'),
    Integer(100, 500, name='epochs')
]

space_model2 = [
    Integer(16, 48, name='latent_dim'),
    Integer(32, 128, name='hidden_dim'),
    Integer(4, 16, name='supervisor_hidden_dim'),
    Categorical([0.01, 0.001, 0.0001], name='lr'),
    Integer(64, 128, name='batch_size'),
    Integer(100, 200, name='epochs')
]

search_spaces = {
    'DirectPred': space_model1,
    'SVAE': space_model2
}

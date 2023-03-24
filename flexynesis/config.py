# config.py
from skopt.space import Integer, Categorical

space_model1 = [
    Integer(16, 128, name='latent_dim'),
    Integer(32, 256, name='hidden_dim'),
    Categorical([0.01, 0.001, 0.0001], name='lr'),
    Integer(32, 64, name='batch_size'),
    Integer(100, 200, name='epochs')
]

search_spaces = {
    'DirectPred': space_model1,
}
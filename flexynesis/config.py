# config.py
from skopt.space import Integer, Categorical, Real

epochs = [200]

search_spaces = {
    'DirectPred': [
        Integer(16, 128, name='latent_dim'),
        Integer(64, 512, name='hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical(epochs, name='epochs')
    ], 
    'supervised_vae': [
        Integer(16, 128, name='latent_dim'),
        Integer(64, 512, name='hidden_dim'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical(epochs, name='epochs')
    ],
    'MultiTripletNetwork': [
        Integer(16, 128, name='latent_dim'),
        Integer(64, 512, name='hidden_dim'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(32, 128, name='batch_size'),
        Categorical(epochs, name='epochs')
    ],
    "DirectPredCNN": [
        Integer(16, 128, name="latent_dim"),
        Integer(64, 512, name="hidden_dim"),
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Integer(32, 128, name="batch_size"),
        Categorical(epochs, name="epochs")
    ],
    "DirectPredGCNN": [
        Integer(16, 128, name="latent_dim"),
        Integer(64, 512, name="hidden_dim"),
        Real(0.0001, 0.01, prior="log-uniform", name="lr"),
        Integer(32, 128, name="batch_size"),
        Categorical(epochs, name="epochs")
    ],
}

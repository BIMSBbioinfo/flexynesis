# config.py
# tuning options for different models
model_tune_config = {
    "DirectPred": {
        "latent_dim": [64, 256], # embedding size for each omics matrix
        "hidden_dim": [64, 256], # hidden_dim for the MLP sub-network
        "lr":  [1e-4, 1e-3, 1e-2], # learning rate 
        "batch_size": [64, 128] # batch size 
    }
}
# default parameters to use, when not tuning 
model_config = {
    "DirectPred": {
        "latent_dim": 64, # embedding size for each omics matrix
        "hidden_dim": 32, # hidden_dim for the MLP sub-network
        "lr":  1e-3, # learning rate 
        "batch_size": 128 # batch size 
    }
}
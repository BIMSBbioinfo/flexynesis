# config.py
model_config = {
    "DirectPred": {
        "latent_dim": [64, 128, 256, 512], # embedding size for each omics matrix
        "hidden_dim": [64, 128, 256], # hidden_dim for the MLP sub-network
        "lr": [1e-4, 1e-2], # learning rate 
        "batch_size": [64] # batch size 
    }
}
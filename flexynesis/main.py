import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

# given a pytorch lightning model and pytorch dataset
def train_model(model, dataset, n_epoch, batch_size, val_size = 0):
    """
    Train a PyTorch Lightning model using the provided dataset.

    This function trains the given model using the provided dataset and training parameters.
    It supports both training without validation and training with validation using a validation set split from the dataset.

    Args:
        model (pl.LightningModule): A PyTorch Lightning model to be trained.
        dataset (torch.utils.data.Dataset): The dataset to be used for training (and validation, if specified).
        n_epoch (int): The number of training epochs.
        batch_size (int): The size of each mini-batch during training (and validation, if specified).
        val_size (float, optional): The fractional size of the validation set relative to the dataset size. Should be between 0 and 1. Defaults to 0 (no validation).

    Returns:
        pl.LightningModule: The trained model.

    Example:

        # Instantiate a model and dataset
        model = MyLightningModel()
        dataset = MyDataset()

        # Train the model for 10 epochs with a batch size of 32 and a validation set comprising 20% of the dataset
        trained_model = train_model(model, dataset, n_epoch=10, batch_size=32, val_size=0.2)
    """
    if val_size == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, limit_val_batches = 0, num_sanity_val_steps = 0, 
                             strategy=DDPStrategy(find_unused_parameters=False), num_nodes = 4) 
        trainer.fit(model, train_loader) 
    elif val_size > 0:
        # split train into train/val
        dat_train, dat_val = random_split(dataset, [1-val_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(dat_train, batch_size=batch_size, num_workers = 0)
        val_loader = DataLoader(dat_val, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, 
                             strategy=DDPStrategy(find_unused_parameters=False), 
                             num_nodes = 4) 
        trainer.fit(model, train_loader, val_loader) 
    return model
    
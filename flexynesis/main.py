import torch 
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from skopt import Optimizer
from skopt.utils import use_named_args
from .config import search_spaces

import numpy as np

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


class HyperparameterTuning:
    def __init__(self, dataset, model_class, config_name, n_epoch = 10, n_iter = 10):
        self.dataset = dataset
        self.model_class = model_class
        self.config_name = config_name
        self.space = search_spaces[config_name]
        self.n_iter = n_iter
        self.n_epoch = n_epoch

    def objective(self, params):
        model = self.model_class(params, self.dataset)
        print(params)
        trainer = pl.Trainer(max_epochs=int(params['epochs']))
        print(trainer)
        # Train and validate the model
        trainer.fit(model)
        val_loss = trainer.validate(model)[0]['val_loss']
        return val_loss    
    
    def perform_tuning(self):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="EI", acq_optimizer="lbfgs")

        best_loss = np.inf
        best_params = None

        for i in range(self.n_iter):
            suggested_params_list = opt.ask()
            suggested_params_dict = {param.name: value for param, value in zip(self.space, suggested_params_list)}
            loss = self.objective(suggested_params_dict)
            opt.tell(suggested_params_list, loss)

            if loss < best_loss:
                best_loss = loss
                best_params = suggested_params_list

        # Update the model-specific configuration with the best hyperparameters
        best_params_dict = {param.name: value for param, value in zip(self.space, best_params)}
        print("Building final model with best params:",best_params_dict)
        # Train the model with the best hyperparameters
        model = self.model_class(best_params_dict, self.dataset)
        trainer = pl.Trainer(max_epochs=int(best_params_dict['epochs']))
        trainer.fit(model)
        return model, best_params
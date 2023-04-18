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


class HyperparameterTuning:
    def __init__(self, dataset, model_class, config_name, task, n_iter = 10):
        self.dataset = dataset
        self.model_class = model_class
        self.config_name = config_name
        self.space = search_spaces[config_name]
        self.n_iter = n_iter
        self.task = task

    def objective(self, params):
        model = self.model_class(params, self.dataset, self.task)
        print(params)
        trainer = pl.Trainer(max_epochs=int(params['epochs']))
        print(trainer)
        # Train and validate the model
        trainer.fit(model)
        val_loss = trainer.validate(model)[0]['val_loss']
        return val_loss    
    
    def perform_tuning(self):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="EI", acq_optimizer="sampling")

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
        model = self.model_class(best_params_dict, self.dataset, self.task)
        trainer = pl.Trainer(max_epochs=int(best_params_dict['epochs']))
        trainer.fit(model)
        return model, best_params
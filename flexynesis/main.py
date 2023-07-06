import torch 
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from tqdm import tqdm

from skopt import Optimizer
from skopt.utils import use_named_args
from .config import search_spaces

import numpy as np



class HyperparameterTuning:
    def __init__(self, dataset, model_class, config_name, target_variables, batch_variables = None, n_iter = 10):
        self.dataset = dataset
        self.model_class = model_class
        self.target_variables = target_variables.strip().split(',')
        self.batch_variables = batch_variables.strip().split(',') if batch_variables is not None else None
        self.config_name = config_name
        self.space = search_spaces[config_name]
        self.n_iter = n_iter
        self.progress_bar = RichProgressBar(
                                theme = RichProgressBarTheme(
                                    progress_bar = 'green1',
                                    metrics = 'yellow', time='gray',
                                    progress_bar_finished='red'))

    def objective(self, params):
        model = self.model_class(params, self.dataset, self.target_variables, self.batch_variables)
        print(params)
        trainer = pl.Trainer(max_epochs=int(params['epochs']), log_every_n_steps=1, 
                            callbacks = self.progress_bar) 
        try:
            # Train the model
            trainer.fit(model)
            # Validate the model
            val_loss = trainer.validate(model)[0]['val_loss_epoch']
        except ValueError as e:
            print(str(e))
            val_loss = float('inf')  # or some other value indicating failure
        return val_loss    
    
    def perform_tuning(self):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="gp_hedge", acq_optimizer="auto")

        best_loss = np.inf
        best_params = None

        with tqdm(total=self.n_iter, desc='Tuning Progress') as pbar:
            for i in range(self.n_iter):
                suggested_params_list = opt.ask()
                suggested_params_dict = {param.name: value for param, value in zip(self.space, suggested_params_list)}
                loss = self.objective(suggested_params_dict)
                opt.tell(suggested_params_list, loss)

                if loss < best_loss:
                    best_loss = loss
                    best_params = suggested_params_list

                # Print result of each iteration
                pbar.set_postfix({'Iteration': i+1, 'Best Loss': best_loss})
                pbar.update(1)
        # Update the model-specific configuration with the best hyperparameters
        best_params_dict = {param.name: value for param, value in zip(self.space, best_params)}
        print("Building final model with best params:",best_params_dict)
        # Train the model with the best hyperparameters
        model = self.model_class(best_params_dict, self.dataset, self.target_variables, self.batch_variables)
        trainer = pl.Trainer(max_epochs=int(best_params_dict['epochs']), callbacks = self.progress_bar)
        trainer.fit(model)
        return model, best_params
    
    
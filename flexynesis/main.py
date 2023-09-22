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

import os, yaml
from skopt.space import Integer, Categorical, Real

            
class HyperparameterTuning:
    def __init__(self, dataset, model_class, config_name, target_variables, batch_variables = None, n_iter = 10, config_path = None, plot_losses = False):
        self.dataset = dataset
        self.model_class = model_class
        self.target_variables = target_variables.strip().split(',')
        self.batch_variables = batch_variables.strip().split(',') if batch_variables is not None else None
        self.config_name = config_name
        self.n_iter = n_iter
        self.plot_losses = plot_losses # Whether to show live loss plots (useful in interactive mode)
        self.progress_bar = RichProgressBar(
                                theme = RichProgressBarTheme(
                                    progress_bar = 'green1',
                                    metrics = 'yellow', time='gray',
                                    progress_bar_finished='red'))
        
        # If config_path is provided, use it
        if config_path:
            external_config = self.load_and_convert_config(config_path)
            if self.config_name in external_config:
                self.space = external_config[self.config_name]
            else:
                raise ValueError(f"'{self.config_name}' not found in the provided config file.")
        else:
            if self.config_name in search_spaces:
                self.space = search_spaces[self.config_name]
            else:
                raise ValueError(f"'{self.config_name}' not found in the default config.")

    def objective(self, params, current_step, total_steps):
        model = self.model_class(params, self.dataset, self.target_variables, self.batch_variables)
        print(params)
        
        mycallbacks = [self.progress_bar]
        # for interactive usage, only show loss plots 
        if self.plot_losses:
            mycallbacks = [LiveLossPlot(hyperparams=params, current_step=current_step, total_steps=total_steps)]      
            
        trainer = pl.Trainer(max_epochs=int(params['epochs']), log_every_n_steps=5, 
                            callbacks = mycallbacks) 
        try:
            # Train the model
            trainer.fit(model)
            # Validate the model
            val_loss = trainer.validate(model)[0]['val_loss']
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
                loss = self.objective(suggested_params_dict, current_step=i+1, total_steps=self.n_iter)
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
        # Train the final model with the best hyperparameters
        final_model = self.model_class(best_params_dict, self.dataset, self.target_variables, self.batch_variables)
        
        mycallbacks = [self.progress_bar]
        # for interactive usage, only show loss plots 
        if self.plot_losses:
            mycallbacks = [LiveLossPlot(hyperparams=best_params_dict, current_step="Final(Best) Step", total_steps=self.n_iter)]      
            
        trainer = pl.Trainer(max_epochs=int(best_params_dict['epochs']), log_every_n_steps=5, callbacks = mycallbacks)
        trainer.fit(final_model)
        return final_model, best_params_dict
    
    def load_and_convert_config(self, config_path):
        # Ensure the config file exists
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file '{config_path}' doesn't exist.")

        # Read the config file
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as file:
                loaded_config = yaml.safe_load(file)
        else:
            raise ValueError("Unsupported file format. Use .yaml or .yml")

        # Convert to skopt space
        search_space_user = {}
        for model, space_definition in loaded_config.items():
            space = []
            for entry in space_definition:
                entry_type = entry.pop("type")
                if entry_type == "Integer":
                    space.append(Integer(**entry))
                elif entry_type == "Real":
                    space.append(Real(**entry))
                elif entry_type == "Categorical":
                    space.append(Categorical(**entry))
                else:
                    raise ValueError(f"Unknown space type: {entry_type}")
            search_space_user[model] = space
        return search_space_user


import matplotlib.pyplot as plt
from IPython.display import clear_output
from pytorch_lightning import Callback

class LiveLossPlot(Callback):
    def __init__(self, hyperparams, current_step, total_steps, figsize=(10, 8)):
        super().__init__()
        self.hyperparams = hyperparams
        self.current_step = current_step
        self.total_steps = total_steps
        self.figsize = figsize

    def on_train_start(self, trainer, pl_module):
        self.losses = {}

    def on_train_end(self, trainer, pl_module):
        plt.ioff()

    def on_train_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(value.item())
        self.plot_losses()

    def plot_losses(self):
        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        for key, losses in self.losses.items():
            plt.plot(losses, label=key)
        
        hyperparams_str = ', '.join(f"{key}={value}" for key, value in self.hyperparams.items())
        title = f"HPO Step={self.current_step} out of {self.total_steps}\n({hyperparams_str})"
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()  # Adjust layout so everything fits
        plt.show()

from lightning import seed_everything
# Set the seed for all the possible random number generators.
seed_everything(42, workers=True)
import torch 
from torch.utils.data import DataLoader, random_split

import lightning as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import EarlyStopping

from tqdm import tqdm

from skopt import Optimizer
from skopt.utils import use_named_args
from .config import search_spaces

import numpy as np

import os, yaml
from skopt.space import Integer, Categorical, Real

            
class HyperparameterTuning:
    """
    Hyperparameter Tuning class for optimizing model parameters.

    This class provides functionalities to perform hyperparameter tuning using Bayesian optimization.
    It supports various features like live loss plotting, early stopping, and custom configuration loading.

    Attributes:
        dataset: Dataset used for training and validation.
        model_class: The class of the model to be tuned.
        target_variables: List of target variables for the model.
        batch_variables: List of batch variables, if applicable.
        config_name: Name of the configuration for tuning parameters.
        n_iter: Number of iterations for the tuning process.
        plot_losses: Boolean flag to plot losses during training.
        val_size: Validation set size as a fraction of the dataset.
        use_loss_weighting: Flag to use loss weighting during training.
        early_stop_patience: Number of epochs to wait for improvement before stopping.
        device_type: Str (cpu, gpu)
    Methods:
        objective(params, current_step, total_steps): Evaluates a set of parameters.
        perform_tuning(): Executes the hyperparameter tuning process.
        init_early_stopping(): Initializes early stopping mechanism.
        load_and_convert_config(config_path): Loads and converts a configuration file.
    """
    def __init__(self, dataset, model_class, config_name, target_variables, 
                 batch_variables = None, surv_event_var = None, surv_time_var = None, 
                 n_iter = 10, config_path = None, plot_losses = False,
                 val_size = 0.2, use_loss_weighting = True, early_stop_patience = -1,
                 device_type = None, gnn_conv_type = None):
        self.dataset = dataset
        self.model_class = model_class
        self.target_variables = target_variables
        self.device_type = device_type
        if self.device_type is None:
            self.device_type = "gpu" if torch.cuda.is_available() else "cpu"
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        self.batch_variables = batch_variables
        self.config_name = config_name
        self.n_iter = n_iter
        self.plot_losses = plot_losses # Whether to show live loss plots (useful in interactive mode)
        self.val_size = val_size
        self.progress_bar = RichProgressBar(
                                theme = RichProgressBarTheme(
                                    progress_bar = 'green1',
                                    metrics = 'yellow', time='gray',
                                    progress_bar_finished='red'))
        self.early_stop_patience = early_stop_patience
        self.use_loss_weighting = use_loss_weighting
        self.gnn_conv_type = gnn_conv_type
        
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
        
        # args common to all model classes 
        model_args = {"config": params, "dataset": self.dataset, "target_variables": self.target_variables,
               "batch_variables": self.batch_variables, "surv_event_var": self.surv_event_var, 
               "surv_time_var": self.surv_time_var, "val_size": self.val_size, 
                "use_loss_weighting": self.use_loss_weighting, "device_type": self.device_type}
        if self.model_class.__name__ == 'DirectPredGCNN':
            model_args["gnn_conv_type"] = self.gnn_conv_type
            
        print(model_args)
        model = self.model_class(**model_args)
        print(params)
        
        mycallbacks = [self.progress_bar]
        # for interactive usage, only show loss plots 
        if self.plot_losses:
            mycallbacks = [LiveLossPlot(hyperparams=params, current_step=current_step, total_steps=total_steps)]
        
        if self.early_stop_patience > 0:
            mycallbacks.append(self.init_early_stopping())
            
        trainer = pl.Trainer(max_epochs=int(params['epochs']), log_every_n_steps=5, 
                            callbacks = mycallbacks, default_root_dir="./", logger=False, 
                             enable_checkpointing=False,
                            devices=1, accelerator=self.device_type) 
        
        # Create a new Trainer instance for validation, ensuring single-device processing
        validation_trainer = pl.Trainer(
            logger=False, 
            enable_checkpointing=False,
            devices=1,  # make sure to a single device for validation
            accelerator=self.device_type
        )
        
        try:
            # Train the model
            trainer.fit(model)
            # Validate the model
            val_loss = validation_trainer.validate(model)[0]['val_loss']
        except ValueError as e:
            print(str(e))
            val_loss = float('inf')  # or some other value indicating failure
        return val_loss, model    
    
    def perform_tuning(self):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="gp_hedge", acq_optimizer="auto")

        best_loss = np.inf
        best_params = None
        best_model = None 

        with tqdm(total=self.n_iter, desc='Tuning Progress') as pbar:
            for i in range(self.n_iter):
                np.int = int
                suggested_params_list = opt.ask()
                suggested_params_dict = {param.name: value for param, value in zip(self.space, suggested_params_list)}
                loss, model = self.objective(suggested_params_dict, current_step=i+1, total_steps=self.n_iter)
                opt.tell(suggested_params_list, loss)

                if loss < best_loss:
                    best_loss = loss
                    best_params = suggested_params_list
                    best_model = model

                # Print result of each iteration
                pbar.set_postfix({'Iteration': i+1, 'Best Loss': best_loss})
                pbar.update(1)
        # convert best params to dict 
        best_params_dict = {param.name: value for param, value in zip(self.space, best_params)}
        return best_model, best_params_dict
    
    def init_early_stopping(self):
        """Initialize the early stopping callback."""
        return EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            verbose=True,
            mode='min'
        )

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
from lightning import Callback

class LiveLossPlot(Callback):
    """
    A callback for visualizing training loss in real-time during hyperparameter optimization.

    This class is a PyTorch Lightning callback that plots training loss and other metrics live as the model trains.
    It is especially useful for tracking the progress of hyperparameter optimization (HPO) steps.

    Attributes:
        hyperparams (dict): Hyperparameters being used in the current HPO step.
        current_step (int): The current step number in the HPO process.
        total_steps (int): The total number of steps in the HPO process.
        figsize (tuple): Size of the figure used for plotting.

    Methods:
        on_train_start(trainer, pl_module): Initializes the loss tracking at the start of training.
        on_train_end(trainer, pl_module): Actions to perform at the end of training.
        on_train_epoch_end(trainer, pl_module): Updates and plots the loss after each training epoch.
        plot_losses(): Renders the loss plot with the current training metrics.
    """
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

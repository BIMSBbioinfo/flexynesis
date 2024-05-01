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
from .data import TripletMultiOmicDataset

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
                 device_type = None, gnn_conv_type = None, 
                 input_layers = None, output_layers = None):
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
        self.input_layers = input_layers
        self.output_layers = output_layers 
        
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
                # get batch sizes (a function of dataset size)
                self.space.append(self.get_batch_space())
            else:
                raise ValueError(f"'{self.config_name}' not found in the default config.")

    def get_batch_space(self, min_size = 16, max_size = 256):
        m = int(np.log2(len(self.dataset) * (1 - self.val_size)))
        st = int(np.log2(min_size))
        end = int(np.log2(max_size))
        if m < end:
            end = m
        s = Categorical([np.power(2, x) for x in range(st, end+1)], name = 'batch_size')
        return s
    
    def objective(self, params, current_step, total_steps):
        
        # args common to all model classes 
        model_args = {"config": params, "dataset": self.dataset, "target_variables": self.target_variables,
               "batch_variables": self.batch_variables, "surv_event_var": self.surv_event_var, 
               "surv_time_var": self.surv_time_var, "val_size": self.val_size, 
                "use_loss_weighting": self.use_loss_weighting, "device_type": self.device_type}
        if self.model_class.__name__ == 'DirectPredGCNN':
            model_args["gnn_conv_type"] = self.gnn_conv_type
        if self.model_class.__name__ == 'CrossModalPred': 
            model_args["input_layers"] = self.input_layers
            model_args["output_layers"] = self.output_layers
            
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


from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
import numpy as np
import random, copy, logging

class FineTuner(pl.LightningModule):
    def __init__(self, model, dataset, n_splits=5, batch_size=32, learning_rates=None, max_epoch = 50, freeze_configs = None):
        super().__init__()
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        self.original_model = model 
        self.dataset = dataset  # Use the entire dataset
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.kfold = KFold(n_splits=self.n_splits, shuffle=True)
        self.learning_rates = learning_rates if learning_rates else [model.config['lr'], model.config['lr']/10, model.config['lr']/100]
        self.folds_data = list(self.kfold.split(np.arange(len(self.dataset))))            
        self.max_epoch = max_epoch
        self.freeze_configs = freeze_configs if freeze_configs else [
                    {'encoders': True, 'supervisors': False},
                    {'encoders': False, 'supervisors': True},
                    {'encoders': False, 'supervisors': False}
                ]
        
        if model.__class__.__name__ == 'MultiTripletNetwork':
            # modify dataset structure to accommodate TripletNetworks
            self.dataset = TripletMultiOmicDataset(dataset, model.main_var)
    
    def apply_freeze_config(self, config):
        # Freeze or unfreeze encoders
        for encoder in self.model.encoders:
            for param in encoder.parameters():
                param.requires_grad = not config['encoders']
                
        # Freeze or unfreeze supervisors
        for mlp in self.model.MLPs.values():
            for param in mlp.parameters():
                param.requires_grad = not config['supervisors']

    def train_dataloader(self):
        # Override to load data for the current fold
        train_idx, val_idx = self.folds_data[self.current_fold]
        train_subset = torch.utils.data.Subset(self.dataset, train_idx)
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # Override to load validation data for the current fold
        train_idx, val_idx = self.folds_data[self.current_fold]
        val_subset = torch.utils.data.Subset(self.dataset, val_idx)
        return DataLoader(val_subset, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx, log=False)

    def validation_step(self, batch, batch_idx):
        # Call the model's validation step without logging
        val_loss = self.model.validation_step(batch, batch_idx, log=False)  
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def run_experiments(self):
        val_loss_results = []
        for lr in self.learning_rates:
            for config in self.freeze_configs:
                fold_losses = []
                epochs = [] # record how many epochs the training happened
                for fold in range(self.n_splits):
                    model_copy = copy.deepcopy(self.original_model)  # Deep copy the model for each fold
                    self.model = model_copy
                    self.apply_freeze_config(config) # try freezing different components 
                    self.current_fold = fold
                    self.learning_rate = lr
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        verbose=False,
                        mode='min'
                    )
                    trainer = pl.Trainer(max_epochs=self.max_epoch, devices=1, accelerator='auto', logger=False, enable_checkpointing=False, 
                                        enable_progress_bar = False, enable_model_summary=False, callbacks=[early_stopping])
                    trainer.fit(self)
                    stopped_epoch = early_stopping.stopped_epoch
                    val_loss = trainer.validate(self.model, verbose = False)
                    fold_losses.append(val_loss[0]['val_loss'])  # Adjust based on your validation output format
                    epochs.append(stopped_epoch)
                    #print(f"[INFO] Finetuning ... training fold: {fold}, learning rate: {lr}, val_loss: {val_loss}, freeze {config}")
                avg_val_loss = np.mean(fold_losses)
                avg_epochs = int(np.mean(epochs))
                print(f"[INFO] average 5-fold cross-validation loss {avg_val_loss} for learning rate: {lr} freeze {config}, average epochs {avg_epochs}")
                val_loss_results.append({'learning_rate': lr, 'average_val_loss': avg_val_loss, 'freeze': config, 'epochs': avg_epochs})

        # Find the best configuration based on validation loss
        best_config = min(val_loss_results, key=lambda x: x['average_val_loss'])
        print(f"Best learning rate: {best_config['learning_rate']} and freeze {best_config['freeze']}", 
              f"with average validation loss: {best_config['average_val_loss']} and average epochs: {best_config['epochs']}")

        # build a final model using the best setup on all samples
        final_model = copy.deepcopy(self.model)
        self.model = final_model
        self.learning_rate = best_config['learning_rate']
        self.apply_freeze_config(best_config['freeze']) 
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        final_trainer = pl.Trainer(max_epochs=best_config['epochs'], devices=1, accelerator='auto', logger=False, enable_checkpointing=False)
        final_trainer.fit(self, train_dataloaders=dl)
        
        
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
    def __init__(self, hyperparams, current_step, total_steps, figsize=(8, 6)):
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
        plt.rcParams['figure.figsize'] = self.figsize
        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        epochs_to_show = 25  # Number of most recent epochs to display

        for key, losses in self.losses.items():
            # If there are more than 25 epochs, slice the list to get the last 25 entries
            if len(losses) > epochs_to_show:
                losses_to_plot = losses[-epochs_to_show:]
                epochs_range = range(len(losses) - epochs_to_show, len(losses))
            else:
                losses_to_plot = losses
                epochs_range = range(len(losses))

            plt.plot(epochs_range, losses_to_plot, label=key)

        hyperparams_str = ', '.join(f"{key}={value}" for key, value in self.hyperparams.items())
        title = f"HPO Step={self.current_step} out of {self.total_steps}\n({hyperparams_str})"

        plt.title(title)
        plt.xlabel(f"Last {epochs_to_show} Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()  # Adjust layout so everything fits
        plt.show()
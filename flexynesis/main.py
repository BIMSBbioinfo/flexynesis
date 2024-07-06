from lightning import seed_everything
seed_everything(42, workers=True)

import torch 
from torch.utils.data import DataLoader, random_split
import torch_geometric

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
from .data import STRING

torch.set_float32_matmul_precision("medium")
            
class HyperparameterTuning:
    """
    A class dedicated to performing hyperparameter tuning using Bayesian optimization for various types of models.
    It supports both cross-validation and single validation set approaches, incorporates early stopping, and allows
    for live loss plotting during the training process for interactive usage. 
    Configuration for the hyperparameter space can be loaded from
    external configuration files in yam format. 
    The class is designed to handle both GPU and CPU environments.

    Attributes:
        dataset: Dataset used for model training and validation.
        model_class: Class of the model for which hyperparameters are being tuned.
        target_variables: List of target variables the model predicts.
        batch_variables: List of batch variables used in model training, if applicable.
        config_name: Configuration name which specifies the hyperparameter search space.
        n_iter: Number of iterations for the Bayesian optimization process.
        plot_losses: Boolean indicating whether to plot losses during training.
        cv_splits: Number of cross-validation splits (if user requested cross-validation)
        use_loss_weighting: Boolean indicating whether to use loss weighting in the model.
        early_stop_patience: Number of epochs with no improvement after which training will be stopped early.
        device_type: Type of device ('gpu' or 'cpu') to be used for training.
        gnn_conv_type: Specific convolution type if using Graph Neural Networks, otherwise None.
        input_layers: Specific input layers for models that require detailed layer setup.
        output_layers: Specific output layers for models that require detailed layer setup.

    Methods:
        __init__(dataset, model_class, config_name, target_variables, batch_variables=None, surv_event_var=None,
                 surv_time_var=None, n_iter=10, config_path=None, plot_losses=False, val_size=0.2, use_cv=False,
                 cv_splits=5, use_loss_weighting=True, early_stop_patience=-1, device_type=None, gnn_conv_type=None,
                 input_layers=None, output_layers=None): Initializes the hyperparameter tuner with specific settings.

        get_batch_space(min_size=16, max_size=256): Determines the batch size search space based on the dataset size.

        setup_trainer(params, current_step, total_steps, full_train=False): Sets up the trainer with appropriate callbacks
            and configurations for either full training or validation based training.

        objective(params, current_step, total_steps, full_train=False): Evaluates a set of parameters to determine the
            performance of the model using the specified parameters.

        perform_tuning(hpo_patience=0): Executes the hyperparameter tuning process, optionally with patience for early
            stopping based on no improvement in performance.

        init_early_stopping(): Initializes the early stopping mechanism to stop training when validation loss does not
            improve for a specified number of epochs.

        load_and_convert_config(config_path): Loads a configuration file and converts it into a format suitable for
            specifying search spaces in Bayesian optimization.
    """

    def __init__(self, dataset, model_class, config_name, target_variables, 
                 batch_variables = None, surv_event_var = None, surv_time_var = None, 
                 n_iter = 10, config_path = None, plot_losses = False,
                 val_size = 0.2,  use_cv = False, cv_splits = 5, 
                 use_loss_weighting = True, early_stop_patience = -1,
                 device_type = None, gnn_conv_type = None, 
                 input_layers = None, output_layers = None):
        self.dataset = dataset # dataset for model initiation
        self.loader_dataset = dataset # dataset for defining data loaders (this can be model specific)
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
        self.use_cv = use_cv
        self.n_splits = cv_splits
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
        
        self.DataLoader = torch.utils.data.DataLoader # use torch data loader by default
        
        if self.model_class.__name__ == 'MultiTripletNetwork':
            self.loader_dataset = TripletMultiOmicDataset(self.dataset, self.target_variables[0])

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

    def get_batch_space(self, min_size = 32, max_size = 256):
        m = int(np.log2(len(self.dataset) * 0.8))
        st = int(np.log2(min_size))
        end = int(np.log2(max_size))
        if m < end:
            end = m
        s = Categorical([np.power(2, x) for x in range(st, end+1)], name = 'batch_size')
        return s
    
    def setup_trainer(self, params, current_step, total_steps, full_train = False):
        # Configure callbacks and trainer for the current fold
        mycallbacks = [self.progress_bar]
        if self.plot_losses:
            mycallbacks.append(LiveLossPlot(hyperparams=params, current_step=current_step, total_steps=total_steps))
        # when training on a full dataset; no cross-validation or no validation splits; 
        # we don't do early stopping
        early_stop_callback = None
        if self.early_stop_patience > 0 and full_train == False:
            early_stop_callback = self.init_early_stopping()
            mycallbacks.append(early_stop_callback)
    
        trainer = pl.Trainer(
            #deterministic = True, 
            precision = '16-mixed', # mixed precision training 
            max_epochs=int(params['epochs']),
            gradient_clip_val=1.0,  
            gradient_clip_algorithm='norm',
            log_every_n_steps=5,
            callbacks=mycallbacks,
            default_root_dir="./",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator=self.device_type
        )
        return trainer, early_stop_callback
    
    def objective(self, params, current_step, total_steps, full_train = False):
        # Unpack or construct specific model arguments
        model_args = {
            "config": params,
            "dataset": self.dataset,
            "target_variables": self.target_variables,
            "batch_variables": self.batch_variables,
            "surv_event_var": self.surv_event_var,
            "surv_time_var": self.surv_time_var,
            "use_loss_weighting": self.use_loss_weighting,
            "device_type": self.device_type,
        }
        
        if self.model_class.__name__ == 'GNN':
            model_args['gnn_conv_type'] = self.gnn_conv_type
        if self.model_class.__name__ == 'CrossModalPred':
            model_args['input_layers'] = self.input_layers
            model_args['output_layers'] = self.output_layers

        if full_train:
            # Train on the full dataset
            full_loader = self.DataLoader(self.loader_dataset, batch_size=int(params['batch_size']), 
                                     shuffle=True, pin_memory=True, drop_last=True)
            model = self.model_class(**model_args)
            trainer, _ = self.setup_trainer(params, current_step, total_steps, full_train = True)
            trainer.fit(model, train_dataloaders=full_loader)
            return model  # Return the trained model

        else:
            validation_losses = []
            epochs = []

            if self.use_cv: # if the user asks for cross-validation
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                split_iterator = kf.split(self.loader_dataset)
            else: # otherwise do a single train/validation split 
                # Compute the number of samples for training based on the ratio
                num_val = int(len(self.loader_dataset) * self.val_size)
                num_train = len(self.loader_dataset) - num_val
                train_subset, val_subset = random_split(self.loader_dataset, [num_train, num_val])
                # create single split format similar to KFold
                split_iterator = [(list(train_subset.indices), list(val_subset.indices))]  
            i = 1
            model = None # save the model if not using cross-validation
            for train_index, val_index in split_iterator:
                print(f"[INFO] {'training cross-validation fold' if self.use_cv else 'training validation split'} {i}")
                train_subset = torch.utils.data.Subset(self.loader_dataset, train_index)
                val_subset = torch.utils.data.Subset(self.loader_dataset, val_index)
                train_loader = self.DataLoader(train_subset, batch_size=int(params['batch_size']), 
                                               pin_memory=True, shuffle=True, drop_last=True, num_workers = 4, prefetch_factor = None, persistent_workers = True)
                val_loader = self.DataLoader(val_subset, batch_size=int(params['batch_size']), 
                                             pin_memory=True, shuffle=False, num_workers = 4, prefetch_factor = None, persistent_workers = True)

                model = self.model_class(**model_args)
                trainer, early_stop_callback = self.setup_trainer(params, current_step, total_steps)
                print(f"[INFO] hpo config:{params}")
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                if early_stop_callback.stopped_epoch:
                    epochs.append(early_stop_callback.stopped_epoch)
                else:
                    epochs.append(int(params['epochs']))
                validation_result = trainer.validate(model, dataloaders=val_loader)
                val_loss = validation_result[0]['val_loss']
                validation_losses.append(val_loss)
                i += 1
                if not self.use_cv:
                    model = model

            # Calculate average validation loss across all folds
            avg_val_loss = np.mean(validation_losses)
            avg_epochs = int(np.mean(epochs))
            return avg_val_loss, avg_epochs, model 
    
    def perform_tuning(self, hpo_patience = 0):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="gp_hedge", acq_optimizer="auto")

        best_loss = np.inf
        best_params = None
        best_epochs = 0
        best_model = None

        # keep track of the streak of HPO iterations without improvement 
        no_improvement_count = 0

        with tqdm(total=self.n_iter, desc='Tuning Progress') as pbar:
            for i in range(self.n_iter):
                np.int = int  # Ensure int type is correctly handled
                suggested_params_list = opt.ask()
                suggested_params_dict = {param.name: value for param, value in zip(self.space, suggested_params_list)}
                loss, avg_epochs, model = self.objective(suggested_params_dict, current_step=i+1, total_steps=self.n_iter)
                if self.use_cv:
                    print(f"[INFO] average 5-fold cross-validation loss {loss} for params: {suggested_params_dict}")
                opt.tell(suggested_params_list, loss)

                if loss < best_loss:
                    best_loss = loss
                    best_params = suggested_params_list
                    best_epochs = avg_epochs
                    best_model = model
                    no_improvement_count = 0  # Reset the no improvement counter
                else:
                    no_improvement_count += 1  # Increment the no improvement counter

                # Print result of each iteration
                pbar.set_postfix({'Iteration': i+1, 'Best Loss': best_loss})
                pbar.update(1)

                # Early stopping condition
                if no_improvement_count >= hpo_patience & hpo_patience > 0:
                    print(f"No improvement in best loss for {hpo_patience} iterations, stopping hyperparameter optimisation early.")
                    break  # Break out of the loop
                best_params_dict = {param.name: value for param, value in zip(self.space, best_params)} if best_params else None
                print(f"[INFO] current best val loss: {best_loss}; best params: {best_params_dict} since {no_improvement_count} hpo iterations")
                

        # Convert best parameters from list to dictionary and include epochs
        best_params_dict = {param.name: value for param, value in zip(self.space, best_params)}
        best_params_dict['epochs'] = best_epochs

        if self.use_cv:
            # Build a final model based on best parameters if using cross-validation
            print(f"[INFO] Building a final model using best params: {best_params_dict}")
            best_model = self.objective(best_params_dict, current_step=0, total_steps=1, full_train=True)

        return best_model, best_params_dict    
    def init_early_stopping(self):
        """Initialize the early stopping callback."""
        return EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            verbose=False,
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
    """
    FineTuner class is designed for fine-tuning trained flexynesis models with flexible control over parameters such as 
    learning rates and component freezing, utilizing cross-validation to optimize generalization.

    This class allows the application of different configuration strategies to either freeze or unfreeze specific 
    model components, while also exploring different learning rates to find the optimal setting. 
    It carries out cross-validation to find the best combination of parameter freezing strategies and learning rates.

    Attributes:
        model (pl.LightningModule): The model instance to be fine-tuned.
        dataset (Dataset): The dataset used for training and validation.
        n_splits (int): Number of cross-validation splits.
        batch_size (int): Batch size for training and validation.
        learning_rates (list): List of learning rates to try during fine-tuning.
        max_epoch (int): Maximum number of epochs for training.
        freeze_configs (list of dicts): Configurations specifying which components of the model to freeze.

    Methods:
        apply_freeze_config(config): Apply a freezing configuration to the model components.
        train_dataloader(): Returns a DataLoader for the training data of the current fold.
        val_dataloader(): Returns a DataLoader for the validation data of the current fold.
        training_step(batch, batch_idx): Executes a training step using the model's internal training logic.
        validation_step(batch, batch_idx): Executes a validation step using the model's internal validation logic.
        configure_optimizers(): Sets up the optimizer with the current learning rate and filtered trainable parameters.
        run_experiments(): Executes the finetuning process across all configurations and learning rates, evaluates
                           using cross-validation, and selects the best configuration based on validation loss.
    """
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
                    trainer.fit(self, train_dataloaders=self.train_dataloader(), val_dataloaders=self.val_dataloader())
                    stopped_epoch = early_stopping.stopped_epoch
                    val_loss = trainer.validate(self, dataloaders = self.val_dataloader(), verbose = False)
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

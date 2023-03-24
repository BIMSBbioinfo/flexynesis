import torch 
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune 
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers.pb2 import PB2

from .config import model_config, model_tune_config


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
    def __init__(self, dataset, model_class, config_name, n_epoch = 10, tune = True, num_cpus = 1, num_gpus=0, val_size=0.2):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)
        if self.tune:
            self.config = self._get_config(config_name)
        else:
            self.config = model_config.get(config_name, {})
        self.best_config = None
        self.hyper_bounds = {key: [min(value), max(value)] for key, value in self.config.items()}
        
    def _get_config(self, config_name):
        config = {}
        model_conf = model_tune_config.get(config_name, {})
        for key, value in model_conf.items():
            config[key] = tune.choice(value)
        return config
    
    def train_model_no_tune(self):
        model = self.model_class(self.config, self.dataset)
        print("Training with parameters:", self.config)
        trainer = pl.Trainer(max_epochs=self.n_epoch)
        trainer.fit(model)
        return model

    def train_model_tune(self):
        model = self.model_class(self.config, self.dataset)
        trainer = Trainer(
            max_epochs=self.n_epoch,
            #gpus=math.ceil(num_gpus),
            logger=TensorBoardLogger(
                save_dir=tune.get_trial_dir(), name="", version="."),
            callbacks=[
                TuneReportCallback(
                    {
                        "loss": "ptl/val_loss",
                        "corr": "ptl/val_corr"
                    },
                    on="validation_end")
            ],
        )
        trainer.fit(model)
        return model

    def tune_model_pb2(self, num_samples=10, n_epoch=10, val_size=0.2):
        scheduler = PB2(
            time_attr='training_iteration',
            perturbation_interval=int(n_epoch/10),
            hyperparam_bounds=self.hyper_bounds
        )

        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"],
            print_intermediate_tables=True
        )

        resources_per_trial = {"cpu": self.num_cpus, "gpu": self.num_gpus}
        
        # Define a wrapper function for train_model_tune
        train_fn_wrapper = lambda *args, **kwargs: self.train_model_tune()

        # Pass the wrapper function to tune.with_parameters
        train_fn_with_parameters = tune.with_parameters(train_fn_wrapper)
        
        analysis = tune.run(train_fn_with_parameters,
                            metric='loss',
                            mode='min',
                            resources_per_trial=resources_per_trial,
                            config=self.config,
                            num_samples=num_samples,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            name="tune_model_pb2")

        self.best_config = analysis.best_config
        print("Best hyperparameters found were: ", self.best_config)
        return self.best_config

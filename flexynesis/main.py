from lightning import seed_everything
seed_everything(42, workers=True)

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
import numpy as np

import lightning as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import EarlyStopping, Callback

from tqdm import tqdm

from skopt import Optimizer
from skopt.utils import use_named_args
from skopt.space import Integer, Categorical, Real

from .config import search_spaces
from .data import TripletMultiOmicDataset
from .inference import run_inference

import os, yaml, copy, logging, random
import matplotlib.pyplot as plt
from IPython.display import clear_output

torch.set_float32_matmul_precision("medium")


def main():
    """
    Main entry point for flexynesis.
    Handles both training and inference-only modes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Flexynesis CLI")
    parser.add_argument("--pretrained_model", type=str, help="Path to pretrained model (.pth)")
    parser.add_argument("--artifacts", type=str, help="Path to preprocessing artifacts (.joblib)")
    parser.add_argument("--data_path_test", type=str, help="Path to test dataset")
    parser.add_argument("--join_key", type=str, help="Join key column in clin.csv")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--prefix", type=str, default="infer", help="Output file prefix")

    args, unknown = parser.parse_known_args()

    # Inference-only mode
    inference_args = [args.pretrained_model, args.artifacts, args.data_path_test]
    if any(inference_args):
        missing = []
        if not args.pretrained_model:
            missing.append("--pretrained_model")
        if not args.artifacts:
            missing.append("--artifacts")
        if not args.data_path_test:
            missing.append("--data_path_test")
        if missing:
            raise SystemExit(f"[ERROR] Inference mode requested but missing required args: {', '.join(missing)}")
        if args.join_key is None:
            print("[WARN] --join_key not provided. If your pipeline merges on clin.csv, this may fail downstream.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_inference(
            pretrained_model_path=args.pretrained_model,
            artifacts_path=args.artifacts,
            test_data_path=args.data_path_test,
            join_key=args.join_key,
            outdir=args.outdir,
            prefix=args.prefix,
            device=device,
        )
        return

    # Training mode (fallback)
    print("[INFO] No inference arguments provided — running in training mode.")
    # TODO: call your original training entrypoint here
    # e.g., from .train import run_training; run_training(args, unknown)


class HyperparameterTuning:
    """
    A class dedicated to performing hyperparameter tuning using Bayesian optimization for various types of models.
    """

    def __init__(self, dataset, model_class, config_name, target_variables,
                 batch_variables=None, surv_event_var=None, surv_time_var=None,
                 n_iter=10, config_path=None, plot_losses=False,
                 val_size=0.2, use_cv=False, cv_splits=5,
                 use_loss_weighting=True, early_stop_patience=-1,
                 device_type=None, gnn_conv_type=None,
                 input_layers=None, output_layers=None, num_workers=2):
        self.dataset = dataset
        self.loader_dataset = dataset
        self.model_class = model_class
        self.target_variables = target_variables
        self.device_type = device_type or ("gpu" if torch.cuda.is_available() else "cpu")
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        self.batch_variables = batch_variables
        self.config_name = config_name
        self.n_iter = n_iter
        self.plot_losses = plot_losses
        self.val_size = val_size
        self.use_cv = use_cv
        self.n_splits = cv_splits
        self.progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                progress_bar="green1",
                metrics="yellow",
                time="gray",
                progress_bar_finished="red",
            )
        )
        self.early_stop_patience = early_stop_patience
        self.use_loss_weighting = use_loss_weighting
        self.gnn_conv_type = gnn_conv_type
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.num_workers = num_workers

        self.DataLoader = torch.utils.data.DataLoader

        if self.model_class.__name__ == "MultiTripletNetwork":
            self.loader_dataset = TripletMultiOmicDataset(self.dataset, self.target_variables[0])

        if config_path:
            external_config = self.load_and_convert_config(config_path)
            if self.config_name in external_config:
                self.space = external_config[self.config_name]
            else:
                raise ValueError(f"'{self.config_name}' not found in the provided config file.")
        else:
            if self.config_name in search_spaces:
                self.space = search_spaces[self.config_name]
                self.space.append(self.get_batch_space())
            else:
                raise ValueError(f"'{self.config_name}' not found in the default config.")

    def get_batch_space(self, min_size=32, max_size=128):
        m = int(np.log2(len(self.dataset) * 0.8))
        st = int(np.log2(min_size))
        end = int(np.log2(max_size))
        if m < end:
            end = m
        return Categorical([np.power(2, x) for x in range(st, end + 1)], name="batch_size")

    def setup_trainer(self, params, current_step, total_steps, full_train=False):
        mycallbacks = []
        if self.plot_losses:
            mycallbacks.append(LiveLossPlot(hyperparams=params, current_step=current_step, total_steps=total_steps))
        else:
            mycallbacks.append(self.progress_bar)

        early_stop_callback = None
        if self.early_stop_patience > 0 and not full_train:
            early_stop_callback = self.init_early_stopping()
            mycallbacks.append(early_stop_callback)

        trainer = pl.Trainer(
            max_epochs=int(params["epochs"]),
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            log_every_n_steps=5,
            callbacks=mycallbacks,
            default_root_dir="./",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator=self.device_type,
        )
        return trainer, early_stop_callback

    def objective(self, params, current_step, total_steps, full_train=False):
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
        if self.model_class.__name__ == "GNN":
            model_args["gnn_conv_type"] = self.gnn_conv_type
        if self.model_class.__name__ == "CrossModalPred":
            model_args["input_layers"] = self.input_layers
            model_args["output_layers"] = self.output_layers

        if full_train:
            full_loader = self.DataLoader(
                self.loader_dataset,
                batch_size=int(params["batch_size"]),
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
            model = self.model_class(**model_args)
            trainer, _ = self.setup_trainer(params, current_step, total_steps, full_train=True)
            trainer.fit(model, train_dataloaders=full_loader)
            return model

        validation_losses, epochs = [], []
        split_iterator = (
            KFold(n_splits=self.n_splits, shuffle=True).split(self.loader_dataset)
            if self.use_cv
            else [
                (
                    list(torch.utils.data.random_split(self.loader_dataset,
                         [len(self.loader_dataset) - int(len(self.loader_dataset) * self.val_size),
                          int(len(self.loader_dataset) * self.val_size)])[0].indices),
                    list(torch.utils.data.random_split(self.loader_dataset,
                         [len(self.loader_dataset) - int(len(self.loader_dataset) * self.val_size),
                          int(len(self.loader_dataset) * self.val_size)])[1].indices),
                )
            ]
        )

        model = None
        for i, (train_index, val_index) in enumerate(split_iterator, start=1):
            print(f"[INFO] {'training cross-validation fold' if self.use_cv else 'training validation split'} {i}")
            train_subset = torch.utils.data.Subset(self.loader_dataset, train_index)
            val_subset = torch.utils.data.Subset(self.loader_dataset, val_index)

            train_loader = self.DataLoader(
                train_subset,
                batch_size=int(params["batch_size"]),
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                prefetch_factor=None,
                persistent_workers=self.num_workers > 0,
            )
            val_loader = self.DataLoader(
                val_subset,
                batch_size=int(params["batch_size"]),
                pin_memory=True,
                shuffle=False,
                num_workers=self.num_workers,
                prefetch_factor=None,
                persistent_workers=self.num_workers > 0,
            )

            model = self.model_class(**model_args)
            trainer, early_stop_callback = self.setup_trainer(params, current_step, total_steps)
            print(f"[INFO] hpo config:{params}")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            stopped = getattr(early_stop_callback, "stopped_epoch", None)
            epochs.append(int(stopped) if stopped is not None else int(params["epochs"]))
            val_loss = trainer.validate(model, dataloaders=val_loader)[0]["val_loss"]
            validation_losses.append(val_loss)

        avg_val_loss = np.mean(validation_losses)
        avg_epochs = int(np.mean(epochs))
        return avg_val_loss, avg_epochs, model

    def perform_tuning(self, hpo_patience=0):
        opt = Optimizer(dimensions=self.space, n_initial_points=10, acq_func="gp_hedge", acq_optimizer="auto")

        best_loss, best_params, best_epochs, best_model = np.inf, None, 0, None
        no_improvement_count = 0

        with tqdm(total=self.n_iter, desc="Tuning Progress") as pbar:
            for i in range(self.n_iter):
                np.int = int
                suggested_params_list = opt.ask()
                suggested_params_dict = {param.name: value for param, value in zip(self.space, suggested_params_list)}
                loss, avg_epochs, model = self.objective(suggested_params_dict, current_step=i + 1, total_steps=self.n_iter)
                if self.use_cv:
                    print(f"[INFO] average {self.n_splits}-fold CV loss {loss} for params: {suggested_params_dict}")
                opt.tell(suggested_params_list, loss)

                if loss < best_loss:
                    best_loss, best_params, best_epochs, best_model = loss, suggested_params_list, avg_epochs, model
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                pbar.set_postfix({"Iteration": i + 1, "Best Loss": best_loss})
                pbar.update(1)

                if hpo_patience > 0 and no_improvement_count >= hpo_patience:
                    print(f"No improvement in best loss for {hpo_patience} iterations, stopping early.")
                    break

                best_params_dict = {param.name: value for param, value in zip(self.space, best_params)} if best_params else None
                print(f"[INFO] current best val loss: {best_loss}; best params: {best_params_dict} since {no_improvement_count} hpo iterations")

        best_params_dict = {param.name: value for param, value in zip(self.space, best_params)}
        best_params_dict["epochs"] = best_epochs

        if self.use_cv:
            print(f"[INFO] Building a final model using best params: {best_params_dict}")
            best_model = self.objective(best_params_dict, current_step=0, total_steps=1, full_train=True)

        return best_model, best_params_dict

    def init_early_stopping(self):
        return EarlyStopping(monitor="val_loss", patience=self.early_stop_patience, verbose=False, mode="min")

    def load_and_convert_config(self, config_path):
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file '{config_path}' doesn't exist.")
        if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
            raise ValueError("Unsupported file format. Use .yaml or .yml")
        with open(config_path, "r") as file:
            loaded_config = yaml.safe_load(file)

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


class LiveLossPlot(Callback):
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
        plt.rcParams["figure.figsize"] = self.figsize
        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        epochs_to_show = 25
        for key, losses in self.losses.items():
            losses_to_plot = losses[-epochs_to_show:]
            epochs_range = range(len(losses) - len(losses_to_plot), len(losses))
            plt.plot(epochs_range, losses_to_plot, label=key)

        hyperparams_str = ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())
        title = f"HPO Step={self.current_step} out of {self.total_steps}\n({hyperparams_str})"
        plt.title(title)
        plt.xlabel(f"Last {epochs_to_show} Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

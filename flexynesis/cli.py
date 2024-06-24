import argparse
import os
import warnings
from typing import Any, NamedTuple, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import models, utils
from .data import DataImporter, MultiOmicDataset, MultiOmicPYGDataset
from .main import HyperparameterTuning
from .models import *


class AvailableModels(NamedTuple):
    # type AvailableModel = ModelClass: Type, ModelConfig: str
    DirectPred: tuple[DirectPred, str] = DirectPred, "DirectPred"
    supervised_vae: tuple[supervised_vae, str] = supervised_vae, "supervised_vae"
    MultiTripletNetwork: tuple[MultiTripletNetwork, str] = MultiTripletNetwork, "MultiTripletNetwork"
    DirectPredGCNN: tuple[DirectPredGCNN, str] = DirectPredGCNN, "DirectPredGCNN"


def main():
    args = parse_cli_args()
    filterwarnings()
    set_num_threads(args.threads)
    validate_paths(args.data_path, args.outdir)

    model_class, config_name = validate_model_arg(args.model_class)
    train_dataset, test_dataset = import_data(
        config_name,
        args.fusion_type,
        args.data_path,
        args.data_types,
        args.log_transform,
        args.features_min,
        args.features_top_percentile,
    )

    model, best_params = tune(
        train_dataset,
        model_class,
        args.target_variables,
        args.batch_variables,
        config_name,
        args.config_path,
        args.hpo_iter,
        args.use_loss_weighting,
        args.early_stop_patience,
        # NOTE: disable for now
        # args.accelerator,
    )
    predictions: dict = predict(model, test_dataset)

    print("Computing model evaluation metrics")
    evaluate_predictions(predictions, test_dataset, args.outdir, args.prefix)

    print("Computing variable importance scores")
    compute_feature_importance(model, args.outdir, args.prefix)

    print("Extracting sample embeddings")
    embeddings_train, embeddings_test = get_sample_embeddings(model, train_dataset, test_dataset)
    print("Saving extracted embeddings")
    save_sample_embeddings(embeddings_train, embeddings_test, args.outdir, args.prefix)

    print("Printing filtered embeddings")
    embeddings_train_filtered, embeddings_test_filtered = filter_embeddings(
        embeddings_train,
        embeddings_test,
        model.target_variables,
        model.batch_variables,
        train_dataset.ann,
        train_dataset.variable_types,
    )
    print("Saving filtered embeddings")
    save_filtered_sample_embeddings(embeddings_train_filtered, embeddings_test_filtered, args.outdir, args.prefix)

    run_baselines(
        args.evaluate_baseline_performance,
        train_dataset,
        test_dataset,
        model.target_variables,
        args.outdir,
        args.prefix,
    )

    print("Saving model")
    save_model(model, args.outdir, args.prefix)


def parse_cli_args() -> argparse.Namespace:
    main_parser = argparse.ArgumentParser(
        description="Flexynesis - Your PyTorch model training interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parent_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parent_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="(Required) Path to the folder with train/test data files.",
    )
    parent_parser.add_argument(
        "--data_types",
        type=str,
        action="extend",
        nargs="+",
        required=True,
        help="(Required) Which omic data matrices to work on, space-separated: e.g. 'gex cnv'",
    )
    parent_parser.add_argument(
        "--target_variables",
        type=str,
        action="extend",
        nargs="+",
        required=True,
        help="(Required) Which variables in 'clin.csv' to use for predictions, space-separated if multiple",
    )
    parent_parser.add_argument(
        "--batch_variables",
        type=str,
        action="extend",
        nargs="+",
        help="(Optional) Which variables in 'clin.csv' to use for data integration / batch correction, space-separated if multiple",
    )
    parent_parser.add_argument(
        "--config_path",
        type=str,
        help="Optional path to an external hyperparameter configuration file in YAML format.",
    )
    parent_parser.add_argument(
        "--fusion_type",
        type=str,
        default="intermediate",
        choices=["early", "intermediate"],
        help="How to fuse the omics layers",
    )
    parent_parser.add_argument(
        "--hpo_iter",
        type=int,
        default=5,
        help="Number of iterations for hyperparameter optimisation",
    )
    parent_parser.add_argument(
        "--features_min",
        type=int,
        default=500,
        help="Minimum number of features to retain after feature selection",
    )
    parent_parser.add_argument(
        "--features_top_percentile",
        type=float,
        default=0.2,
        help="Top percentile features to retain after feature selection",
    )
    parent_parser.add_argument(
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Path to the output folder to save the model outputs",
    )
    parent_parser.add_argument(
        "--prefix",
        type=str,
        default="job",
        help="Job prefix to use for output files",
    )
    parent_parser.add_argument(
        "--log_transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to apply log-transformation to input data matrices.",
    )
    parent_parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Number of threads to use",
    )
    parent_parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=-1,
        help="How many epochs to wait when no improvements in validation loss is observed (default: -1; no early stopping)",
    )
    parent_parser.add_argument(
        "--loss_weighting",
        dest="use_loss_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to apply loss-balancing using uncertainty weights method.",
    )
    parent_parser.add_argument(
        "--baseline_performance_evaluation",
        dest="evaluate_baseline_performance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset.",
    )
    # NOTE: disable for now
    # parent_parser.add_argument(
    #     "--accelerator",
    #     type=str,
    #     default="auto",
    #     choices=["auto", "cpu", "gpu"],
    #     help="Accelerator type used by pl.Trainer.",
    # )

    subparsers = main_parser.add_subparsers(title="Models", dest="model_class", help="Available Models.")
    model_parsers = {
        model_name: subparsers.add_parser(
            name=model_name,
            parents=[parent_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help=model_name,
        )
        for model_name in models.__all__
    }

    model_parsers["DirectPredGCNN"].add_argument(
        "--graph_file",
        type=str,
        help="Optional path to a graph file.",
    )

    return main_parser.parse_args()


def filterwarnings() -> None:
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", "has been removed as a dependency of the")
    warnings.filterwarnings("ignore", "The `srun` command is available on your system but is not used")


def set_num_threads(threads: int) -> None:
    # Ignore this line when number of threads = -1
    if threads > 0:
        torch.set_num_threads(threads)


def validate_paths(data_path: str, outdir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError("Input --data_path doesn't exist at: {data_path}".format(data_path=data_path))
    if not os.path.exists(outdir):
        raise FileNotFoundError("Path to --outdir doesn't exist at: {outdir}".format(outdir=outdir))


def validate_model_arg(model_class: str) -> tuple[Type, str]:
    available_models = AvailableModels()
    model_class = getattr(available_models, model_class, None)
    if model_class is None:
        raise ValueError(f"Invalid model_class: {model_class}")
    else:
        return model_class


def import_data(
    config_name: str,
    fusion_type: str,
    data_path: str,
    data_types: list[str],
    log_transform: bool,
    features_min: Optional[int],
    features_top_percentile: Optional[float],
    variance_threshold: float = 1e-5,
    na_threshold: float = 0.1,
    use_graph: bool = False,
    node_name: str = "gene_name",
    transform: Optional[Any] = None,
) -> Union[tuple[MultiOmicDataset, MultiOmicDataset], tuple[MultiOmicPYGDataset, MultiOmicPYGDataset]]:
    # Set use_graph var
    use_graph = True if config_name == "DirectPredGCNN" else False
    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    concatenate = False
    if fusion_type == "early":
        concatenate = True
    data_importer = DataImporter(
        path=data_path,
        data_types=data_types,
        concatenate=concatenate,
        log_transform=log_transform,
        min_features=features_min,
        top_percentile=features_top_percentile,
        use_graph=use_graph,
    )
    return data_importer.import_data()


def tune(
    train_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset],
    model_class: Type,
    target_variables: list[str],
    batch_variables: list[str],
    config_name: str,
    config_path: str,
    hpo_iter: int,
    use_loss_weighting: bool,
    early_stop_patience: int,
    # NOTE: disable for now
    # accelerator: str,
) -> tuple[nn.Module, dict]:
    # define a tuner object, which will instantiate a DirectPred class
    # using the input dataset and the tuning configuration from the config.py
    tuner = HyperparameterTuning(
        train_dataset,
        model_class=model_class,
        target_variables=target_variables,
        batch_variables=batch_variables,
        config_name=config_name,
        config_path=config_path,
        n_iter=hpo_iter,
        use_loss_weighting=use_loss_weighting,
        early_stop_patience=early_stop_patience,
    )
    # do a hyperparameter search training multiple models and get the best_configuration
    # NOTE: disable for now
    # return tuner.perform_tuning(accelerator=accelerator)
    return tuner.perform_tuning()


def predict(model: nn.Module, test_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset]) -> dict[str, np.array]:
    # make predictions on the test dataset
    return model.predict(test_dataset)


def evaluate_predictions(predictions, test_dataset, outdir, prefix) -> None:
    metrics_df = utils.evaluate_wrapper(predictions, test_dataset)
    metrics_df.to_csv(os.path.join(outdir, ".".join([prefix, "stats.csv"])), header=True, index=False)


def compute_feature_importance(model: nn.Module, outdir: str, prefix: str) -> None:
    for var in model.target_variables:
        model.compute_feature_importance(var, steps=20)
    df_imp = pd.concat([model.feature_importances[x] for x in model.target_variables], ignore_index=True)
    df_imp.to_csv(os.path.join(outdir, ".".join([prefix, "feature_importance.csv"])), header=True, index=False)


def get_sample_embeddings(
    model: nn.Module,
    train_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset],
    test_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embeddings_train = model.transform(train_dataset)
    embeddings_test = model.transform(test_dataset)
    return embeddings_train, embeddings_test


def save_sample_embeddings(
    embeddings_train: pd.DataFrame, embeddings_test: pd.DataFrame, outdir: str, prefix: str
) -> None:
    # TODO: This will be a generic fn.
    embeddings_train.to_csv(os.path.join(outdir, ".".join([prefix, "embeddings_train.csv"])), header=True)
    embeddings_test.to_csv(os.path.join(outdir, ".".join([prefix, "embeddings_test.csv"])), header=True)


def filter_embeddings(
    embeddings_train: pd.DataFrame,
    embeddings_test: pd.DataFrame,
    target_variables: list[str],
    batch_variables: Optional[list[str]],
    ann: dict[str, torch.Tensor],
    variable_types: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # also filter embeddings to remove batch-associated dims and only keep target-variable associated dims
    batch_dict = {x: ann[x] for x in batch_variables} if batch_variables is not None else None
    target_dict = {x: ann[x] for x in target_variables}
    embeddings_train_filtered = utils.remove_batch_associated_variables(
        data=embeddings_train,
        batch_dict=batch_dict,
        target_dict=target_dict,
        variable_types=variable_types,
    )
    # filter test embeddings to keep the same dims as the filtered training embeddings
    embeddings_test_filtered = embeddings_test[embeddings_train_filtered.columns]
    return embeddings_train_filtered, embeddings_test_filtered


def save_filtered_sample_embeddings(
    embeddings_train_filtered: pd.DataFrame,
    embeddings_test_filtered: pd.DataFrame,
    outdir: str,
    prefix: str,
) -> None:
    embeddings_train_filtered.to_csv(
        os.path.join(outdir, ".".join([prefix, "embeddings_train.filtered.csv"])), header=True
    )
    embeddings_test_filtered.to_csv(
        os.path.join(outdir, ".".join([prefix, "embeddings_test.filtered.csv"])), header=True
    )


def run_baselines(
    eval_baseline_performance: bool,
    train_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset],
    test_dataset: Union[MultiOmicDataset, MultiOmicPYGDataset],
    target_variables: list[str],
    outdir: str,
    prefix: str,
) -> None:
    """Evaluate off-the-shelf methods on the main target variable."""
    if eval_baseline_performance:
        print("Evaluating Baseline models.")
        print("Computing off-the-shelf method performance on first target variable:", target_variables[0])
        metrics_baseline = utils.evaluate_baseline_performance(
            train_dataset, test_dataset, variable_name=target_variables[0], n_folds=5
        )
        metrics_baseline.to_csv(
            os.path.join(outdir, ".".join([prefix, "baseline.stats.csv"])), header=True, index=False
        )
    else:
        print("Skipping Baseline models.")


def save_model(model, outdir: str, prefix: str):
    """Save the trained model in file."""
    torch.save(model, os.path.join(outdir, ".".join([prefix, "final_model.pth"])))


if __name__ == "__main__":
    main()

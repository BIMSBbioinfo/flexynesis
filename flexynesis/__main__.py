import os
import sys
import argparse
import yaml
import time
import random
import warnings
import json
import tracemalloc
import psutil
from . import __version__

os.environ["OMP_NUM_THREADS"] = "1"


def print_test_installation():
    print("Test Installation:")
    print("  # Download and extract test dataset")
    print("  curl -L -o dataset1.tgz https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dataset1.tgz")
    print("  tar -xzvf dataset1.tgz")
    print()
    print("  # Test the installation (should finish within a minute on a typical CPU)")
    print("  flexynesis --data_path dataset1 --model_class DirectPred --target_variables Erlotinib --hpo_iter 1 --features_top_percentile 5 --data_types gex,cnv")


def print_help():
    print("usage: flexynesis [-h] --data_path DATA_PATH --model_class {DirectPred,supervised_vae,MultiTripletNetwork,CrossModalPred,GNN,RandomForest,SVM,XGBoost,RandomSurvivalForest} --data_types DATA_TYPES")
    print()
    print("Flexynesis model training interface")
    print()
    print("options:")
    print("  -h, --help            show complete help with all options")
    print("  --data_path DATA_PATH")
    print("                        (Required) Path to the folder with train/test data files")
    print("  --model_class {DirectPred,supervised_vae,MultiTripletNetwork,CrossModalPred,GNN,RandomForest,SVM,XGBoost,RandomSurvivalForest}")
    print("                        (Required) The kind of model class to instantiate")
    print("  --data_types DATA_TYPES")
    print("                        (Required) Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'")
    print("  --hpo_iter HPO_ITER   Number of iterations for hyperparameter optimisation (default: 100)")
    print("  --device {auto,cuda,mps,cpu}")
    print("                        Device type: 'auto' (automatic detection), 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon), 'cpu' (default: auto)")
    print("  --use_gpu             (Optional) DEPRECATED: Use --device instead. If set, attempts to use CUDA/GPU if available.")
    print()
    print_test_installation()
    print()
    print("  See the documentation for more details at https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/.")


def print_full_help():
    print("usage: flexynesis [-h] --data_path DATA_PATH --model_class {DirectPred,supervised_vae,MultiTripletNetwork,CrossModalPred,GNN,RandomForest,SVM,XGBoost,RandomSurvivalForest} "
          "[--gnn_conv_type {GC,GCN,SAGE}] [--target_variables TARGET_VARIABLES] [--covariates COVARIATES] [--surv_event_var SURV_EVENT_VAR] [--surv_time_var SURV_TIME_VAR] "
          "[--config_path CONFIG_PATH] [--fusion_type {early,intermediate}] [--hpo_iter HPO_ITER] [--finetuning_samples FINETUNING_SAMPLES] "
          "[--variance_threshold VARIANCE_THRESHOLD] [--correlation_threshold CORRELATION_THRESHOLD] [--restrict_to_features RESTRICT_TO_FEATURES] "
          "[--subsample SUBSAMPLE] [--features_min FEATURES_MIN] [--features_top_percentile FEATURES_TOP_PERCENTILE] --data_types DATA_TYPES "
          "[--input_layers INPUT_LAYERS] [--output_layers OUTPUT_LAYERS] [--outdir OUTDIR] [--prefix PREFIX] [--log_transform {True,False}] "
          "[--early_stop_patience EARLY_STOP_PATIENCE] [--hpo_patience HPO_PATIENCE] [--val_size VAL_SIZE] [--use_cv] [--use_loss_weighting {True,False}] "
          "[--evaluate_baseline_performance] [--threads THREADS] [--num_workers NUM_WORKERS] [--device {auto,cuda,mps,cpu}] [--use_gpu] [--feature_importance_method {IntegratedGradients,GradientShap,Both}] "
          "[--disable_marker_finding] [--string_organism STRING_ORGANISM] [--string_node_name {gene_name,gene_id}] [--safetensors]")
    print()
    print("Flexynesis model training interface")
    print()
    print("options:")

    # --- NEW: inference-only flags (keep in full help) ---
    print("  --pretrained_model PRETRAINED_MODEL")
    print("                        Use a saved .pth model for inference (skip training)")
    print("  --artifacts ARTIFACTS")
    print("                        Path to training-time artifacts .joblib")
    print("  --data_path_test DATA_PATH_TEST")
    print("                        Folder with test-only dataset for inference")
    print("  --join_key JOIN_KEY   Column name in 'clin.csv' for sample IDs")

    # --- existing flags (keep full list) ---
    print("  -h, --help            show this help message and exit")
    print("  --data_path DATA_PATH")
    print("                        (Required) Path to the folder with train/test data files")
    print("  --model_class {DirectPred,supervised_vae,MultiTripletNetwork,CrossModalPred,GNN,RandomForest,SVM,XGBoost,RandomSurvivalForest}")
    print("                        (Required) The kind of model class to instantiate")
    print("  --gnn_conv_type {GC,GCN,SAGE}")
    print("                        If model_class is set to GNN, choose which graph convolution type to use")
    print("  --target_variables TARGET_VARIABLES")
    print("                        (Optional if survival variables are not set to None). Which variables in 'clin.csv' to use for predictions, comma-separated if multiple")
    print("  --covariates COVARIATES")
    print("                        Which variables in 'clin.csv' to be used as feature covariates, comma-separated if multiple")
    print("  --surv_event_var SURV_EVENT_VAR")
    print("                        Which column in 'clin.csv' to use as event/status indicator for survival modeling")
    print("  --surv_time_var SURV_TIME_VAR")
    print("                        Which column in 'clin.csv' to use as time/duration indicator for survival modeling")
    print("  --config_path CONFIG_PATH")
    print("                        Optional path to an external hyperparameter configuration file in YAML format.")
    print("  --fusion_type {early,intermediate}")
    print("                        How to fuse the omics layers (default: intermediate)")
    print("  --hpo_iter HPO_ITER   Number of iterations for hyperparameter optimisation (default: 100)")
    print("  --finetuning_samples FINETUNING_SAMPLES")
    print("                        Number of samples from the test dataset to use for fine-tuning the model. Set to 0 to disable fine-tuning (default: 0)")
    print("  --variance_threshold VARIANCE_THRESHOLD")
    print("                        Variance threshold (as percentile) to drop low variance features (default is 1; set to 0 for no variance filtering)")
    print("  --correlation_threshold CORRELATION_THRESHOLD")
    print("                        Correlation threshold to drop highly redundant features (default is 0.8; set to 1 for no redundancy filtering)")
    print("  --restrict_to_features RESTRICT_TO_FEATURES")
    print("                        Restrict the analysis to the list of features provided by the user (default is None)")
    print("  --subsample SUBSAMPLE")
    print("                        Downsample training set to randomly drawn N samples for training. Disabled when set to 0 (default: 0)")
    print("  --features_min FEATURES_MIN")
    print("                        Minimum number of features to retain after feature selection (default: 500)")
    print("  --features_top_percentile FEATURES_TOP_PERCENTILE")
    print("                        Top percentile features (among the features remaining after variance filtering and data cleanup) to retain after feature selection (default: 20)")
    print("  --data_types DATA_TYPES")
    print("                        (Required) Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'")
    print("  --input_layers INPUT_LAYERS")
    print("                        If model_class is set to CrossModalPred, choose which data types to use as input/encoded layers. Comma-separated if multiple")
    print("  --output_layers OUTPUT_LAYERS")
    print("                        If model_class is set to CrossModalPred, choose which data types to use as output/decoded layers. Comma-separated if multiple")
    print("  --outdir OUTDIR       Path to the output folder to save the model outputs (default: current working directory)")
    print("  --prefix PREFIX       Job prefix to use for output files (default: 'job')")
    print("  --log_transform {True,False}")
    print("                        whether to apply log-transformation to input data matrices (default: False)")
    print("  --early_stop_patience EARLY_STOP_PATIENCE")
    print("                        How many epochs to wait when no improvements in validation loss is observed (default 10; set to -1 to disable early stopping)")
    print("  --hpo_patience HPO_PATIENCE")
    print("                        How many hyperparameter optimisation iterations to wait for when no improvements are observed (default is 10; set to 0 to disable early stopping)")
    print("  --val_size VAL_SIZE   Proportion of training data to be used as validation split (default: 0.2)")
    print("  --use_cv              (Optional) If set, a 5-fold cross-validation training will be done. Otherwise, a single training on 80 percent of the dataset is done.")
    print("  --use_loss_weighting {True,False}")
    print("                        whether to apply loss-balancing using uncertainty weights method (default: True)")
    print("  --evaluate_baseline_performance")
    print("                        whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset")
    print("  --threads THREADS     (Optional) How many threads to use when using CPU (default is 4)")
    print("  --num_workers NUM_WORKERS")
    print("                        (Optional) How many workers to use for model training (default is 0)")
    print("  --device {auto,cuda,mps,cpu}")
    print("                        Device type: 'auto' (automatic detection), 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon), 'cpu' (default: auto)")
    print("  --use_gpu             (Optional) DEPRECATED: Use --device instead. If set, attempts to use CUDA/GPU if available.")
    print("  --feature_importance_method {IntegratedGradients,GradientShap,Both}")
    print("                        Choose feature importance score method (default: IntegratedGradients)")
    print("  --disable_marker_finding")
    print("                        (Optional) If set, marker discovery after model training is disabled.")
    print("  --string_organism STRING_ORGANISM")
    print("                        STRING DB organism id. (default: 9606)")
    print("  --string_node_name {gene_name,gene_id}")
    print("                        Type of node name. (default: gene_name)")
    print("  --safetensors         If set, the model will be saved in the SafeTensors format. Default is False.")
    print()
    print_test_installation()
    print()
    print("  See the documentation for more details at https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/.")


def main():
    """
    Parse command-line arguments and run the Flexynesis training/inference interface.

    This entry point configures argument parsing for training and evaluating PyTorch
    models (and baselines), including data paths, model selection, hyperparameters,
    and runtime options. It also supports a pure **inference mode** that skips training.

    Args:
      --data_path (str):
        Path to the folder with train/test data files. **Required** unless running
        in inference mode with `--data_path_test`.

      --model_class (str):
        Model class to instantiate. One of:
        ["DirectPred", "GNN", "supervised_vae", "MultiTripletNetwork",
         "CrossModalPred", "RandomForest", "SVM", "XGBoost", "RandomSurvivalForest"].
        **Required** for training.

      --gnn_conv_type (str):
        If `--model_class=GNN`, choose graph convolution: ["GC", "GCN", "SAGE"].

      --target_variables (str):
        Comma-separated target variables from `clin.csv`. Optional if survival
        variables are provided.

      --surv_event_var (str):
        Column in `clin.csv` used as event/status indicator (survival).

      --surv_time_var (str):
        Column in `clin.csv` used as time/duration indicator (survival).

      --config_path (str):
        Optional path to a YAML hyperparameter configuration file.

      --fusion_type (str):
        Omics fusion strategy: ["early", "intermediate"]. Default: `intermediate`.

      --hpo_iter (int):
        Number of hyperparameter optimization iterations. Default: `100`.

      --finetuning_samples (int):
        Number of test samples used for fine-tuning; `0` disables. Default: `0`.

      --variance_threshold (float):
        Variance percentile threshold to drop low-variance features. Default: `1`.
        Set `0` to disable.

      --correlation_threshold (float):
        Correlation threshold to drop redundant features. Default: `0.8`.
        Set `1` to disable.

      --restrict_to_features (str):
        Path to a user-provided feature list to restrict analysis. Default: `None`.

      --subsample (int):
        Randomly downsample training to N samples; `0` disables. Default: `0`.

      --features_min (int):
        Minimum features to retain after selection. Default: `500`.

      --features_top_percentile (float):
        Keep top percentile (after filtering/cleanup). Default: `20`.

      --data_types (str):
        Comma-separated omics matrices to use (e.g., `gex,cnv`). **Required**.

      --input_layers (str):
        For `CrossModalPred`: input/encoded data types (comma-separated).

      --output_layers (str):
        For `CrossModalPred`: output/decoded data types (comma-separated).

      --outdir (str):
        Output directory. Default: current working directory.

      --prefix (str):
        Job prefix for output files. Default: `job`.

      --log_transform (str):
        Apply log-transform to input matrices: ["True", "False"]. Default: `False`.

      --early_stop_patience (int):
        Epochs to wait without val-loss improvement. Default: `10`. Use `-1` to disable.

      --hpo_patience (int):
        HPO iterations to wait without improvement. Default: `10`. Use `0` to disable.

      --use_cv (bool):
        If set, perform 5-fold cross-validation; otherwise train once on 80% split.

      --val_size (float):
        Fraction of training data used for validation. Default: `0.2`.

      --use_loss_weighting (str):
        Apply uncertainty-based loss balancing: ["True", "False"]. Default: `True`.

      --evaluate_baseline_performance (bool):
        Also train baselines (RandomForest, SVM) for comparison.

      --threads (int):
        CPU thread count. Default: `4`.

      --num_workers (int):
        Data-loading workers for training. Default: `0`.

      --use_gpu (bool):
        **Deprecated** — use `--device` instead.

      --device (str):
        Training device: ["auto", "cuda", "mps", "cpu"]. Default: `auto`.
        `auto` selects the best available device.

      --feature_importance_method (str):
        Feature importance method(s): ["IntegratedGradients", "GradientShap", "Both"].
        Default: `Both`.

      --disable_marker_finding (bool):
        Disable marker discovery post-training.

      --string_organism (int):
        STRING DB organism ID. Default: `9606`.

      --string_node_name (str):
        Node name type: ["gene_name", "gene_id"]. Default: `gene_name`.

      --safetensors (bool):
        Save the model in SafeTensors format. Default: `False`.

    Inference mode (skip training):
    
      To run *inference only*, provide **all three** of the following:
      
        --pretrained_model (str): Path to a saved model file (e.g., `model.pth` or `.safetensors`).
        --artifacts (str): Path to the training artifacts bundle (e.g., `artifacts.joblib`).
        --data_path_test (str): Folder containing test-only data.

      Optional:
      
        --join_key (str): Column in `clin.csv` used to join samples. Default: `JoinKey`.

      Behavior:
        When the three required arguments above are present, the CLI:
        
          1) skips training and hyperparameter search,
          2) loads the pretrained model and artifacts,
          3) performs inference on `--data_path_test`,
          4) writes predictions/metrics to `--outdir` with the chosen `--prefix`,
          5) exits.

    Examples:
    
      Train:
      
        ```
        flexynesis --data_path ./data --data_types gex,cnv --model_class DirectPred 
        ```

      Inference only:
      
        ```
        flexynesis --pretrained_model ./outputs/best_model.pth --artifacts ./outputs/artifacts.joblib --data_path_test ./data_test --outdir ./predictions --prefix run1
        ```
        
    """

    # Early help (no heavy imports)
    if len(sys.argv) == 1:
        print_help()
        return
    if any(arg in ['-h', '--help'] for arg in sys.argv):
        print_full_help()
        return

    # ------------- Parser (lightweight) -------------
    parser = argparse.ArgumentParser(
        description="Flexynesis model training interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    # Existing core flags  (made not-required here; enforced conditionally below)
    parser.add_argument("--data_path", type=str, required=False,
                        help="Path to the folder with train/test data files")
    parser.add_argument("--model_class", type=str, required=False,
                        choices=["DirectPred", "supervised_vae", "MultiTripletNetwork", "CrossModalPred", "GNN", "RandomForest", "SVM", "XGBoost", "RandomSurvivalForest"],
                        help="The kind of model class to instantiate")
    parser.add_argument("--gnn_conv_type", type=str, choices=["GC", "GCN", "SAGE"],
                        help="If model_class is set to GNN, choose which graph convolution type to use")
    parser.add_argument("--target_variables", type=str, default=None,
                        help="(Optional if survival variables are not set to None). Which variables in 'clin.csv' to use for predictions, comma-separated if multiple")
    parser.add_argument("--covariates", type=str, default=None,
                        help="Which variables in 'clin.csv' to be used as feature covariates, comma-separated if multiple")
    parser.add_argument("--surv_event_var", type=str, default=None,
                        help="Which column in 'clin.csv' to use as event/status indicator for survival modeling")
    parser.add_argument("--surv_time_var", type=str, default=None,
                        help="Which column in 'clin.csv' to use as time/duration indicator for survival modeling")
    parser.add_argument('--config_path', type=str, default=None,
                        help='Optional path to an external hyperparameter configuration file in YAML format.')
    parser.add_argument("--fusion_type", type=str, choices=["early", "intermediate"], default='intermediate',
                        help="How to fuse the omics layers")
    parser.add_argument("--hpo_iter", type=int, default=100,
                        help="Number of iterations for hyperparameter optimisation")
    parser.add_argument("--finetuning_samples", type=int, default=0,
                        help="Number of samples from the test dataset to use for fine-tuning the model. Set to 0 to disable fine-tuning")
    parser.add_argument("--variance_threshold", type=float, default=1,
                        help="Variance threshold (as percentile) to drop low variance features (default is 1; set to 0 for no variance filtering)")
    parser.add_argument("--correlation_threshold", type=float, default=0.8,
                        help="Correlation threshold to drop highly redundant features (default is 0.8; set to 1 for no redundancy filtering)")
    parser.add_argument("--restrict_to_features", type=str, default=None,
                        help="Restrict the analyis to the list of features provided by the user (default is None)")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Downsample training set to randomly drawn N samples for training. Disabled when set to 0")
    parser.add_argument("--features_min", type=int, default=500,
                        help="Minimum number of features to retain after feature selection")
    parser.add_argument("--features_top_percentile", type=float, default=20,
                        help="Top percentile features (among the features remaining after variance filtering and data cleanup) to retain after feature selection")
    parser.add_argument("--data_types", type=str, required=False,
                        help="Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'")
    parser.add_argument("--input_layers", type=str, default=None,
                        help="If model_class is set to CrossModalPred, choose which data types to use as input/encoded layers. Comma-separated if multiple")
    parser.add_argument("--output_layers", type=str, default=None,
                        help="If model_class is set to CrossModalPred, choose which data types to use as output/decoded layers. Comma-separated if multiple")
    parser.add_argument("--outdir", type=str, default=os.getcwd(),
                        help="Path to the output folder to save the model outputs")
    parser.add_argument("--prefix", type=str, default='job',
                        help="Job prefix to use for output files")
    parser.add_argument("--log_transform", type=str, choices=['True', 'False'], default='False',
                        help="whether to apply log-transformation to input data matrices")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="How many epochs to wait when no improvements in validation loss is observed (default 10; set to -1 to disable early stopping)")
    parser.add_argument("--hpo_patience", type=int, default=20,
                        help="How many hyperparamater optimisation iterations to wait for when no improvements are observed (default is 10; set to 0 to disable early stopping)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion of training data to be used as validation split (default: 0.2)")
    parser.add_argument("--use_cv", action="store_true",
                        help="(Optional) If set, the a 5-fold cross-validation training will be done. Otherwise, a single trainig on 80 percent of the dataset is done.")
    parser.add_argument("--use_loss_weighting", type=str, choices=['True', 'False'], default='True',
                        help="whether to apply loss-balancing using uncertainty weights method")
    parser.add_argument("--evaluate_baseline_performance", action="store_true",
                        help="whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset")
    parser.add_argument("--threads", type=int, default=4,
                        help="(Optional) How many threads to use when using CPU (default is 4)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="(Optional) How many workers to use for model training (default is 0)")
    parser.add_argument("--use_gpu", action="store_true", 
                        help="(Optional) DEPRECATED: Use --device instead. If set, attempts to use CUDA/GPU if available.")
    parser.add_argument("--device", type=str, 
                        choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Device type: 'auto' (automatic detection), 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon), 'cpu'")
    parser.add_argument("--feature_importance_method", type=str,
                        choices=["IntegratedGradients", "GradientShap", "Both"], default="IntegratedGradients",
                        help="Choose feature importance score method")
    parser.add_argument("--disable_marker_finding", action="store_true",
                        help="(Optional) If set, marker discovery after model training is disabled.")
    # GNN args.
    parser.add_argument("--string_organism", type=int, default=9606,
                        help="STRING DB organism id.")
    parser.add_argument("--string_node_name", type=str, choices=["gene_name", "gene_id"], default="gene_name",
                        help="Type of node name.")
    # safetensors args
    parser.add_argument("--safetensors", action="store_true",
                        help="If set, the model will be saved in the SafeTensors format. Default is False.")
    # NEW: inference flags
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to a saved model (.pth) to use for inference")
    parser.add_argument("--artifacts", type=str, default=None,
                        help="Path to artifacts .joblib saved during training")
    parser.add_argument("--data_path_test", type=str, default=None,
                        help="Folder with test-only dataset for inference")
    parser.add_argument("--join_key", type=str, default="JoinKey",
                        help="Column name in 'clin.csv' (test metadata) used to join sample IDs")

    args = parser.parse_args()

    # --------- Conditional requirements & I/O prep (applies to both modes) ---------
    # Ensure outdir exists (works for training and inference)
    if not args.outdir:
        args.outdir = os.getcwd()
    os.makedirs(args.outdir, exist_ok=True)

    # Only require core training flags if NOT doing inference
    in_infer = bool(args.pretrained_model)
    if not in_infer:
        missing = [k for k in ("data_path", "model_class", "data_types") if not getattr(args, k, None)]
        if missing:
            parser.error("the following arguments are required in training mode: " +
                         ", ".join(f"--{m}" for m in missing))

    # ---------- Inference mode: early exit path ----------
    if args.pretrained_model and args.artifacts and args.data_path_test:
        import torch
        from .inference import run_inference
        from .utils import get_optimal_device, create_device_from_string

        # quick existence checks
        if not os.path.exists(args.pretrained_model):
            raise FileNotFoundError(f"--pretrained_model not found: {args.pretrained_model}")
        if not os.path.exists(args.artifacts):
            raise FileNotFoundError(f"--artifacts not found: {args.artifacts}")

        # Handle device selection for inference (same logic as training)
        if args.use_gpu:
            warnings.warn("--use_gpu is deprecated. Use --device instead.", DeprecationWarning)
            if args.device != "auto":
                device_preference = args.device
                print(f"[WARN] Both --use_gpu and --device {args.device} specified. Using --device {args.device}.")
            else:
                # Let auto-detection find the best GPU device (CUDA or MPS)
                device_preference = "auto"
        else:
            device_preference = args.device
        
        # Get optimal device for inference
        device_str, device_type = get_optimal_device(device_preference)
        device = create_device_from_string(device_str)
        print(f"[INFO] Using device for inference: {device_str}")

        # Robust load across PyTorch versions & checkpoint types
        try:
            model = torch.load(args.pretrained_model, map_location=device, weights_only=False)
        except TypeError:
            model = torch.load(args.pretrained_model, map_location=device)
        except Exception:
            try:
                model = torch.load(args.pretrained_model, map_location=device, weights_only=True)
            except TypeError:
                model = torch.load(args.pretrained_model, map_location=device)

        if hasattr(model, "to"):
            model.to(device).eval()

        run_inference(
            model=model,
            artifacts_path=args.artifacts,
            data_path_test=args.data_path_test,
            outdir=args.outdir,
            prefix=args.prefix,
        )
        return  # inside main(); done after inference

    # ------------- Heavy imports only when training -------------
    print("[INFO] Loading Flexynesis modules...")
    import flexynesis
    from lightning import seed_everything
    import lightning as pl
    from typing import NamedTuple
    import torch
    import pandas as pd
    from safetensors.torch import save_file

    # models
    from .models.direct_pred import DirectPred
    from .models.supervised_vae import supervised_vae
    from .models.triplet_encoder import MultiTripletNetwork
    from .models.crossmodal_pred import CrossModalPred
    from .models.gnn_early import GNN

    # data + utils
    from .data import STRING, MultiOmicDatasetNW, DataImporter
    from .main import HyperparameterTuning, FineTuner
    from .utils import evaluate_baseline_performance, evaluate_baseline_survival_performance, get_predicted_labels, evaluate_wrapper, get_optimal_device, get_device_memory_info, create_device_from_string
    import tracemalloc, psutil
    import json

    # --------- Sanity checks on args ---------
    # 1. survival variables consistency
    if (args.surv_event_var is None) != (args.surv_time_var is None):
        parser.error("Both --surv_event_var and --surv_time_var must be provided together or left as None.")

    # 2. required variables for model classes
    if args.model_class not in ("supervised_vae", "CrossModalPred"):
        if not any([args.target_variables, args.surv_event_var]):
            parser.error("When selecting a model other than 'supervised_vae' or 'CrossModalPred', you must provide at least one of --target_variables, or survival variables (--surv_event_var and --surv_time_var)")

    # 3. Check for compatibility of fusion_type with CrossModalPred
    if args.fusion_type == "early":
        if args.model_class == 'CrossModalPred':
            parser.error("The 'CrossModalPred' model cannot be used with early fusion type. "
                         "Use --fusion_type intermediate instead.")
            
    
    # 4. Handle device selection with MPS support
    # Support legacy --use_gpu flag for backward compatibility
    if args.use_gpu:
        warnings.warn("--use_gpu is deprecated. Use --device instead.", DeprecationWarning)
        # If --device is not explicitly set (still at default auto), let auto-detection handle it
        if args.device != "auto":
            # If both --use_gpu and explicit --device are provided, respect --device but warn
            device_preference = args.device
            print(f"[WARN] Both --use_gpu and --device {args.device} specified. Using --device {args.device}.")
        else:
            # Let auto-detection find the best GPU device (CUDA or MPS)
            device_preference = "auto"
    else:
        device_preference = args.device
    
    # Get optimal device using new device detection
    device_str, device_type = get_optimal_device(device_preference)
    
    # Print device information
    print(f"[INFO] Using device: {device_str}")
    if device_str != 'cpu':
        memory_info = get_device_memory_info(device_str)
        print(f"[INFO] Device name: {memory_info['device_name']}")
        if device_str == 'cuda':
            print(f"[INFO] Available CUDA devices: {memory_info['device_count']}")

    # gnn
    if args.model_class == 'GNN':
        if not args.gnn_conv_type:
            warnings.warn("\n\n!!! When running GNN, set --gnn_conv_type (GC/GCN/SAGE). Falling back to GC !!!\n")
            time.sleep(3)
            gnn_conv_type = 'GC'
        else:
            gnn_conv_type = args.gnn_conv_type
    else:
        gnn_conv_type = None

    # CrossModalPred IO layers
    input_layers = args.input_layers
    output_layers = args.output_layers
    datatypes = args.data_types.strip().split(',')
    if args.model_class == 'CrossModalPred':
        if args.input_layers:
            input_layers = input_layers.strip().split(',')
            if not all(layer in datatypes for layer in input_layers):
                raise ValueError(f"Input layers {input_layers} are not a valid subset of the data types: ({datatypes}).")
        if args.output_layers:
            output_layers = output_layers.strip().split(',')
            if not all(layer in datatypes for layer in output_layers):
                raise ValueError(f"Output layers {output_layers} are not a valid subset of the data types: ({datatypes}).")

    # paths
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Input --data_path doesn't exist at: {args.data_path}")
    if not os.path.exists(args.outdir):
        raise FileNotFoundError(f"Path to --outdir doesn't exist at: {args.outdir}")

    available_models = {
        "DirectPred": (DirectPred, "DirectPred"),
        "supervised_vae": (supervised_vae, "supervised_vae"),
        "MultiTripletNetwork": (MultiTripletNetwork, "MultiTripletNetwork"),
        "CrossModalPred": (CrossModalPred, "CrossModalPred"),
        "GNN": (GNN, "GNN"),
        "RandomForest": ("RandomForest", None),
        "XGBoost": ("XGBoost", None),
        "SVM": ("SVM", None),
        "RandomSurvivalForest": ("RandomSurvivalForest", None),
    }

    model_info = available_models.get(args.model_class)
    if model_info is None:
        raise ValueError(f"Unsupported model class {args.model_class}")
    model_class, config_name = model_info

    # Set concatenate to True to use early fusion, otherwise intermediate
    concatenate = args.fusion_type == 'early' and args.model_class != 'GNN'

    # covariates
    if args.covariates:
        if args.model_class == 'GNN':  # Covariates not yet supported for GNNs
            warnings.warn("\n\n!!! Covariates are currently not supported for GNN models, they will be ignored. !!!\n")
            time.sleep(3)
            covariates = None
        else:
            covariates = args.covariates.strip().split(',')
    else:
        covariates = None

    data_importer = DataImporter(
        path=args.data_path,
        data_types=datatypes,
        covariates=covariates,
        concatenate=concatenate,
        log_transform=args.log_transform == 'True',
        variance_threshold=args.variance_threshold / 100,
        correlation_threshold=args.correlation_threshold,
        restrict_to_features=args.restrict_to_features,
        min_features=args.features_min,
        top_percentile=args.features_top_percentile,
        processed_dir='_'.join(['processed', args.prefix]),
        downsample=args.subsample
    )

    # import data
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    t1 = time.time()
    train_dataset, test_dataset = data_importer.import_data()

    data_import_time = time.time() - t1
    data_import_ram = process.memory_info().rss

    # classical ML baselines
    if args.model_class in ["RandomForest", "SVM", "XGBoost"]:
        if args.target_variables:
            var = args.target_variables.strip().split(',')[0]
            print(f"Training {args.model_class} on variable: {var}")
            metrics, predictions = evaluate_baseline_performance(
                train_dataset, test_dataset, variable_name=var, methods=[args.model_class], n_folds=5, n_jobs=args.threads
            )
            metrics.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'stats.csv'])), header=True, index=False)
            predictions.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'predicted_labels.csv'])), header=True, index=False)
            print(f"{args.model_class} evaluation complete. Results saved.")
            sys.exit(0)
        else:
            raise ValueError("At least one target variable is required to run RandomForest/SVM/XGBoost models. Set --target_variables")

    if args.model_class == "RandomSurvivalForest":
        if args.surv_event_var and args.surv_time_var:
            print(f"Training {args.model_class} on survival variables: {args.surv_event_var} and {args.surv_time_var}")
            metrics, predictions = evaluate_baseline_survival_performance(
                train_dataset, test_dataset, args.surv_time_var, args.surv_event_var, n_folds=5, n_jobs=int(args.threads)
            )
            metrics.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'stats.csv'])), header=True, index=False)
            predictions.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'predicted_labels.csv'])), header=True, index=False)
            print(f"{args.model_class} evaluation complete. Results saved.")
            sys.exit(0)
        else:
            raise ValueError("Missing survival variables. Set --surv_event_var --surv_time_var")

    # GNN overlay (temporary solution)
    if args.model_class == 'GNN':
        print("[INFO] Overlaying the dataset with network data from STRINGDB]")
        obj = STRING(os.path.join(args.data_path, '_'.join(['processed', args.prefix])),
                     args.string_organism, args.string_node_name)
        train_dataset = MultiOmicDatasetNW(train_dataset, obj.graph_df)
        train_dataset.print_stats()
        test_dataset = MultiOmicDatasetNW(test_dataset, obj.graph_df)

    # feature logs
    feature_logs = data_importer.feature_logs
    for key in feature_logs.keys():
        feature_logs[key].to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'feature_logs', key, 'csv'])),
                                 header=True, index=False)

    # tuner
    tuner = HyperparameterTuning(
        dataset=train_dataset,
        model_class=model_class,
        target_variables=args.target_variables.strip().split(',') if args.target_variables is not None else [],
        batch_variables=None,
        surv_event_var=args.surv_event_var,
        surv_time_var=args.surv_time_var,
        config_name=config_name,
        config_path=args.config_path,
        n_iter=int(args.hpo_iter),
        use_loss_weighting=args.use_loss_weighting == 'True',
        val_size=args.val_size,
        use_cv=args.use_cv,
        early_stop_patience=int(args.early_stop_patience),
        device_type=device_type,
        gnn_conv_type=gnn_conv_type,
        input_layers=input_layers,
        output_layers=output_layers,
        num_workers=args.num_workers
    )

    # do a hyperparameter search training multiple models and get the best configuration
    t1 = time.time()
    model, best_params = tuner.perform_tuning(hpo_patience=args.hpo_patience)
    hpo_time = time.time() - t1
    hpo_system_ram = process.memory_info().rss

    # if fine-tuning is enabled; fine tune the model on a portion of test samples
    if args.finetuning_samples > 0:
        finetuneSampleN = args.finetuning_samples
        print("[INFO] Finetuning the model on ", finetuneSampleN, "test samples")
        # split test dataset into finetuning and holdout datasets
        all_indices = range(len(test_dataset))
        import random as _random
        finetune_indices = _random.sample(list(all_indices), finetuneSampleN)
        holdout_indices = list(set(all_indices) - set(finetune_indices))
        finetune_dataset = test_dataset.subset(finetune_indices)
        holdout_dataset = test_dataset.subset(holdout_indices)

        # fine tune on the finetuning dataset; freeze the encoders
        finetuner = FineTuner(model, finetune_dataset)
        finetuner.run_experiments()

        # update the model to finetuned model
        model = finetuner.model
        # update the test dataset to exclude finetuning samples
        test_dataset = holdout_dataset

    # get sample embeddings and save
    print("[INFO] Extracting sample embeddings")
    embeddings_train = model.transform(train_dataset)
    embeddings_test = model.transform(test_dataset)
    embeddings_train.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_train.csv'])), header=True)
    embeddings_test.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_test.csv'])), header=True)

    # evaluate predictions;  (if any supervised learning happened)
    if any([args.target_variables, args.surv_event_var]):
        if not args.disable_marker_finding:  # unless marker discovery is disabled
            # compute feature importance values
            if args.feature_importance_method == 'Both':
                explainers = ['IntegratedGradients', 'GradientShap']
            else:
                explainers = [args.feature_importance_method]

            for explainer in explainers:
                print("[INFO] Computing variable importance scores using explainer:", explainer)
                for var in model.target_variables:
                    model.compute_feature_importance(train_dataset, var, steps_or_samples=25, method=explainer)
                import pandas as pd
                df_imp = pd.concat([model.feature_importances[x] for x in model.target_variables], ignore_index=True)
                df_imp['explainer'] = explainer
                df_imp.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'feature_importance', explainer, 'csv'])), header=True, index=False)

        # print known/predicted labels
        predicted_labels = pd.concat([
            get_predicted_labels(model.predict(train_dataset), train_dataset, 'train', args.model_class),
            get_predicted_labels(model.predict(test_dataset), test_dataset, 'test', args.model_class)
        ], ignore_index=True)
        predicted_labels.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'predicted_labels.csv'])), header=True, index=False)

        print("[INFO] Computing model evaluation metrics")
        metrics_df = evaluate_wrapper(
            args.model_class, model.predict(test_dataset), test_dataset,
            surv_event_var=model.surv_event_var,
            surv_time_var=model.surv_time_var
        )
        metrics_df.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'stats.csv'])), header=True, index=False)

    # for architectures with decoders; print decoded output layers
    if args.model_class == 'CrossModalPred':
        print("[INFO] Printing decoded output layers")
        output_layers_train = model.decode(train_dataset)
        output_layers_test = model.decode(test_dataset)
        for layer in output_layers_train.keys():
            output_layers_train[layer].to_csv(
                os.path.join(args.outdir, '.'.join([args.prefix, 'train_decoded', layer, 'csv'])),
                header=True
            )
        for layer in output_layers_test.keys():
            output_layers_test[layer].to_csv(
                os.path.join(args.outdir, '.'.join([args.prefix, 'test_decoded', layer, 'csv'])),
                header=True
            )

    # evaluate off-the-shelf methods on the main target variable
    if args.evaluate_baseline_performance:
        print("[INFO] Computing off-the-shelf method performance on first target variable:",model.target_variables[0])
        var = model.target_variables[0]
        metrics = pd.DataFrame()

        # in the case when GNNEarly was used, the we use the initial multiomicdataset for train/test
        # because GNNEarly requires a modified dataset structure to fit the networks (temporary solution)
        if args.model_class == 'GNN':
            train = getattr(train_dataset, 'multiomic_dataset', train_dataset)
            test = getattr(test_dataset, 'multiomic_dataset', test_dataset)
        else:
            train = train_dataset
            test = test_dataset
        
        if var != model.surv_event_var: 
            metrics, predictions = evaluate_baseline_performance(train, test, 
                                                               variable_name = var, 
                                                               methods = ['RandomForest', 'SVM', 'XGBoost'],
                                                               n_folds = 5,
                                                               n_jobs = int(args.threads))
            predictions.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'baseline.predicted_labels.csv'])), header=True, index=False)

        if model.surv_event_var and model.surv_time_var:
            print("[INFO] Computing off-the-shelf method performance on survival variable:",model.surv_time_var)
            metrics_baseline_survival = evaluate_baseline_survival_performance(train, test,
                                                                                             model.surv_time_var,
                                                                                             model.surv_event_var,
                                                                                             n_folds = 5,
                                                                                             n_jobs = int(args.threads))
            metrics = pd.concat([metrics, metrics_baseline_survival], axis = 0, ignore_index = True)

        if not metrics.empty:
            metrics.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'baseline.stats.csv'])), header=True, index=False)

    # save the trained model in file
    if not args.safetensors:
        torch.save(model, os.path.join(args.outdir, '.'.join([args.prefix, 'final_model.pth'])))
    else:
        save_file(model.state_dict(), os.path.join(args.outdir, '.'.join([args.prefix, 'final_model.safetensors'])))
        # save model config as JSON
        config = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
        }
        # Common attributes to save
        common_attrs = [
            'input_dims', 'layers',
            'device_type', 'target_variables',
            'surv_event_var', 'surv_time_var',
            'config', 'current_epoch', 'num_layers'
        ]
        for attr in common_attrs:
            if hasattr(model, attr):
                config[attr] = getattr(model, attr)
        if hasattr(model, 'layers'):
            config['num_layers'] = len(model.layers)

        if hasattr(model, 'config'):
            model_specific_config = model.config
            config.update(model_specific_config)

        with open(os.path.join(args.outdir, '.'.join([args.prefix, 'final_model_config.json'])), 'w') as f:
            json.dump(config, f, indent=2, default=str)

    # --- write inference artifacts joblib (auto-generated after training) ---
    try:
        from .inference import InferenceArtifacts
        import joblib  # noqa: F401  (ensures joblib is present if InferenceArtifacts uses it)

        # Build feature list dictionary from the processed training dataset if available
        feature_lists = {}
        if hasattr(train_dataset, "data"):
            try:
                for k, df in getattr(train_dataset, "data", {}).items():
                    try:
                        feature_lists[k] = list(df.columns)
                    except Exception:
                        feature_lists[k] = []
            except Exception:
                feature_lists = {dt: [] for dt in args.data_types.split(',')}
        else:
            feature_lists = {dt: [] for dt in args.data_types.split(',')}

        art = InferenceArtifacts(
            schema_version=1,
            data_types=args.data_types.split(','),
            target_variables=(args.target_variables.split(',') if args.target_variables else []),
            feature_lists=feature_lists,
            transforms={},          # TODO: plug in real scalers/encoders when available
            label_encoders={},      # TODO: plug in label encoders if used
            join_key=args.join_key, # default "JoinKey"
        )
        joblib_path = os.path.join(args.outdir, '.'.join([args.prefix, 'artifacts.joblib']))
        art.save(joblib_path)
        print(f"[INFO] Wrote inference artifacts to {joblib_path}")
    except Exception as e:
        print(f"[WARN] Could not write inference artifacts: {e}")

    print(f"[INFO] Time spent in data import: {data_import_time:.2f} sec")
    print(f"[INFO] RAM after data import: {data_import_ram / (1024**2):.2f} MB")
    print(f"[INFO] Time spent in HPO: {hpo_time:.2f} sec")
    
    # Enhanced device memory reporting
    final_memory_info = get_device_memory_info(device_str)
    if device_str == 'cuda':
        print(f"[INFO] Peak CUDA RAM allocated: {final_memory_info['max_allocated']:.2f} MB")
    elif device_str == 'mps':
        print(f"[INFO] MPS device used (detailed memory tracking not available)")
    
    print(f"[INFO] CPU RAM after HPO: {hpo_system_ram / (1024**2):.2f} MB")
    print("[INFO] Finished the analysis!")


if __name__ == "__main__":
    main()

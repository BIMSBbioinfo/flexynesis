"""
inference.py — reconstruct a full Flexynesis model from safetensors + config + artifacts.
Called by __main__.py when --pretrained_model points to a .safetensors file.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import joblib
import torch
from safetensors.torch import load_file


MODEL_REGISTRY = {
    "DirectPred":        ("flexynesis.models.direct_pred",    "DirectPred"),
    "supervised_vae":    ("flexynesis.models.supervised_vae", "supervised_vae"),
    "CrossModalPred":    ("flexynesis.models.crossmodal_pred","CrossModalPred"),
    "MultiTripletNetwork": ("flexynesis.models.triplet_encoder","MultiTripletNetwork"),
    "GNN":               ("flexynesis.models.gnn_early",      "GNN"),
}


def _import_model_class(model_class_name):
    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {model_class_name}. "
            f"Supported: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[model_class_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_dataset_namespace(config, artifacts):
    """
    Build a pickle-safe SimpleNamespace that satisfies model __init__ signatures.
    Uses joblib artifacts (our native format).
    """
    feature_lists = artifacts.get("feature_lists", {})
    target_vars   = config.get("target_variables", [])
    label_encoders = artifacts.get("label_encoders", {})

    # layers: prefer input_layers from config, fall back to feature_lists keys
    layers = (
        config.get("input_layers")
        or config.get("layers")
        or artifacts.get("original_modalities")
        or artifacts.get("data_types")
        or list(feature_lists.keys())
    )
    if not layers:
        raise ValueError("Cannot infer model layers from config or artifacts.")

    # dat: keys model __init__ reads via dataset.dat.keys()
    dat = {layer: None for layer in layers}

    # variable_types + ann from label_encoders
    variable_types = {}
    ann = {}
    for var, enc in (label_encoders or {}).items():
        if enc is None:
            continue
        cats = enc.categories_[0].tolist() if hasattr(enc, "categories_") else (enc.get("categories", [[]])[0] if isinstance(enc, dict) else [])
        if cats:
            ann[var] = cats
            variable_types[var] = "categorical"

    for var in target_vars:
        if var not in variable_types:
            variable_types[var] = "numerical"
            ann[var] = np.array([0.0])

    return SimpleNamespace(
        layers=layers,
        features=feature_lists,
        dat=dat,
        variable_types=variable_types,
        ann=ann,
    )


def _resolve_input_dims(config, artifacts):
    """Ensure input_dims is present in config, deriving from feature_lists if needed."""
    feature_lists = artifacts.get("feature_lists", {})
    layers = (
        config.get("input_layers")
        or config.get("layers")
        or list(feature_lists.keys())
    )
    input_dims = config.get("input_dims")
    if not input_dims:
        input_dims = [len(feature_lists[l]) for l in layers if l in feature_lists]
        config["input_dims"] = input_dims
    return config


def _load_artifacts(artifacts_path):
    import json
    import joblib

    def check_file_type(file_path):
        with open(file_path, 'rb') as f:
            header = f.read(10)
        try:
            text_header = header.decode('utf-8').lstrip()
            if text_header.startswith('{') or text_header.startswith('['):
                return "json"
        except UnicodeDecodeError:
            pass
        joblib_magic_bytes = (b'\x80', b'\x1f\x8b', b'BZh', b'\x04"M\x18', b'\x78', b'\xfd7zXZ')
        if header.startswith(joblib_magic_bytes):
            return "joblib"
        return "unknown"

    file_type = check_file_type(artifacts_path)
    if file_type == "json":
        with open(artifacts_path, "r") as f:
            return json.load(f)
    elif file_type == "joblib":
        return joblib.load(artifacts_path)
    else:
        raise ValueError(f"[ERROR] The artifacts file {artifacts_path} is neither a valid JSON nor a recognized Joblib format.")


def reconstruct_model(safetensors_path, config_path, artifacts_path, device="cpu"):
    """
    Reconstruct a full Flexynesis model from:
      - safetensors_path : .safetensors weights file
      - config_path      : final_model_config.json
      - artifacts_path   : .artifacts.joblib
      - device           : torch device string

    Returns a fully instantiated, weights-loaded, eval-mode model.
    """
    print(f"[INFO] Reconstructing model from safetensors")
    print(f"[INFO]   config    : {config_path}")
    print(f"[INFO]   artifacts : {artifacts_path}")
    print(f"[INFO]   weights   : {safetensors_path}")

    # 1. Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_class_name = config.get("model_class")
    if not model_class_name:
        raise ValueError("Config JSON is missing 'model_class' field.")
    print(f"[INFO]   model_class: {model_class_name}")

    # 2. Load artifacts
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    artifacts = _load_artifacts(artifacts_path)

    # 3. Resolve dims
    config = _resolve_input_dims(config, artifacts)

    # 4. Import model class
    ModelClass = _import_model_class(model_class_name)

    # 5. Build minimal dataset namespace
    dataset = _build_dataset_namespace(config, artifacts)

    # 6. Extract hyperparams — coerce string ints/floats
    model_config = dict(config.get("config", {}))
    for key in ["latent_dim", "supervisor_hidden_dim", "batch_size"]:
        if key in model_config and isinstance(model_config[key], str):
            model_config[key] = int(model_config[key])

    # 7. Instantiate model
    init_kwargs = dict(
        config=model_config,
        dataset=dataset,
        target_variables=config.get("target_variables", []),
        batch_variables=None,
        surv_event_var=config.get("surv_event_var"),
        surv_time_var=config.get("surv_time_var"),
        use_loss_weighting=True,
        device_type=config.get("device_type", "cpu"),
    )
    # CrossModalPred also takes input_layers / output_layers
    if model_class_name == "CrossModalPred":
        if config.get("input_layers"):
            init_kwargs["input_layers"] = config["input_layers"]
        if config.get("output_layers"):
            init_kwargs["output_layers"] = config["output_layers"]

    model = ModelClass(**init_kwargs)

    # 8. Load weights
    state_dict = load_file(safetensors_path, device=device)
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()

    print(f"[INFO] Model reconstructed successfully: {type(model).__name__}")
    print(f"[INFO]   has .transform(): {hasattr(model, 'transform')}")
    print(f"[INFO]   has .predict()  : {hasattr(model, 'predict')}")
    return model

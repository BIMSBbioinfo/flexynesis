"""
inference.py — reconstruct a full Flexynesis model from safetensors + config + artifacts.
Called by __main__.py when --pretrained_model points to a .safetensors file.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

MODEL_REGISTRY = {
    "DirectPred": ("flexynesis.models.direct_pred", "DirectPred"),
    "supervised_vae": ("flexynesis.models.supervised_vae", "supervised_vae"),
    "CrossModalPred": ("flexynesis.models.crossmodal_pred", "CrossModalPred"),
    "MultiTripletNetwork": (
        "flexynesis.models.triplet_encoder",
        "MultiTripletNetwork",
    ),
    "GNN": ("flexynesis.models.gnn_early", "GNN"),
}


def check_model_type(file_path):
    import json
    import struct

    with open(file_path, "rb") as f:
        header_start = f.read(8)

    if len(header_start) < 8:
        return "unknown"

    # 1. Try SafeTensors check first
    try:
        header_size = struct.unpack("<Q", header_start)[0]
        if header_size < 100_000_000:
            with open(file_path, "rb") as f:
                f.seek(8)
                header_bytes = f.read(header_size)
            header_json = json.loads(header_bytes.decode("utf-8"))
            if isinstance(header_json, dict):
                return "safetensors"
    except (struct.error, UnicodeDecodeError, json.JSONDecodeError):
        pass

    # 2. Try PyTorch ZIP or Pickle check
    # PyTorch models are either ZIP (starts with PK\x03\x04) or Pickle
    # Pickle files start with 0x80 followed by a protocol byte (e.g., 0x02 to 0x05)
    if header_start.startswith(b"PK\x03\x04"):
        return "pth"
    if header_start[0] == 0x80 and header_start[1] in (2, 3, 4, 5):
        return "pth"

    return "unknown"


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
    target_vars = config.get("target_variables", [])
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
        cats = (
            enc.categories_[0].tolist()
            if hasattr(enc, "categories_")
            else (enc.get("categories", [[]])[0] if isinstance(enc, dict) else [])
        )
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
        config.get("input_layers") or config.get("layers") or list(feature_lists.keys())
    )
    input_dims = config.get("input_dims")
    if not input_dims:
        input_dims = [len(feature_lists[l]) for l in layers if l in feature_lists]
        config["input_dims"] = input_dims
    return config


def load_and_sniff_artifacts(artifacts_path):
    """
    Checks if an artifacts file is JSON or Joblib,
    loads it appropriately, and returns (file_type, content).
    """
    import json

    import joblib

    def check_file_type(file_path):
        with open(file_path, "rb") as f:
            header = f.read(10)
        try:
            text_header = header.decode("utf-8").lstrip()
            if text_header.startswith("{") or text_header.startswith("["):
                return "json"
        except UnicodeDecodeError:
            pass
        joblib_magic_bytes = (
            b"\x80",  # pickle binary protocol marker
            b"\x1f\x8b",  # gzip
            b"BZh",  # bzip2
            b'\x04"M\x18',  # LZ4 frame magic
            b"\x78\x9c",  # zlib (deflate)
            b"\x78\xda",  # zlib (deflate, alternate)
            b"\xfd7zXZ",  # xz
        )
        if header.startswith(joblib_magic_bytes):
            return "joblib"
        return "unknown"

    file_type = check_file_type(artifacts_path)
    if file_type == "json":
        with open(artifacts_path, "r") as f:
            raw = json.load(f)
        return "json", raw
    elif file_type == "joblib":
        return "joblib", joblib.load(artifacts_path)
    else:
        raise ValueError(
            f"[ERROR] The artifacts file {artifacts_path} is neither a valid JSON nor a recognized Joblib format."
        )


def _deserialize_json_artifacts(artifacts):
    import numpy as np
    from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                       StandardScaler)

    # Rebuild sklearn objects expected by inference code.
    deserialized = dict(artifacts)

    transforms = {}
    for modality, scaler_dict in artifacts.get("transforms", {}).items():
        if scaler_dict is None:
            transforms[modality] = None
            continue
        scaler_type = scaler_dict.get("type")
        if scaler_type != "StandardScaler":
            raise ValueError(
                f"Unsupported scaler type in artifacts JSON for '{modality}': {scaler_type}"
            )

        scaler = StandardScaler(
            with_mean=scaler_dict.get("with_mean", True),
            with_std=scaler_dict.get("with_std", True),
        )
        if "mean" in scaler_dict:
            scaler.mean_ = np.array(scaler_dict["mean"], dtype=float)
        if "scale" in scaler_dict:
            scaler.scale_ = np.array(scaler_dict["scale"], dtype=float)
        if "var" in scaler_dict:
            scaler.var_ = np.array(scaler_dict["var"], dtype=float)
        if "n_features_in" in scaler_dict:
            scaler.n_features_in_ = int(scaler_dict["n_features_in"])
        if "feature_names_in" in scaler_dict:
            scaler.feature_names_in_ = np.array(
                scaler_dict["feature_names_in"], dtype=object
            )
        if "n_samples_seen" in scaler_dict:
            n_samples_seen = scaler_dict["n_samples_seen"]
            scaler.n_samples_seen_ = (
                np.array(n_samples_seen)
                if isinstance(n_samples_seen, list)
                else int(n_samples_seen)
            )

        transforms[modality] = scaler

    label_encoders = {}
    for variable, encoder_dict in artifacts.get("label_encoders", {}).items():
        if encoder_dict is None:
            label_encoders[variable] = None
            continue

        encoder_type = encoder_dict.get("type")
        if encoder_type == "LabelEncoder":
            enc = LabelEncoder()
            enc.classes_ = np.array(encoder_dict.get("classes", []), dtype=object)
            label_encoders[variable] = enc
            continue

        if encoder_type == "OrdinalEncoder":
            categories = [
                np.array(cat, dtype=object)
                for cat in encoder_dict.get("categories", [])
            ]
            encoded_missing = encoder_dict.get("encoded_missing_value", np.nan)
            if encoded_missing == "__NaN__":
                encoded_missing = np.nan

            # encoded_missing_value is version-dependent in sklearn; fall back.
            ordinal_kwargs = {
                "categories": categories,
                "handle_unknown": encoder_dict.get("handle_unknown", "error"),
                "unknown_value": encoder_dict.get("unknown_value", None),
            }
            try:
                enc = OrdinalEncoder(
                    encoded_missing_value=encoded_missing,
                    **ordinal_kwargs,
                )
            except TypeError:
                enc = OrdinalEncoder(**ordinal_kwargs)
            setattr(enc, "categories_", categories)
            if "encoded_missing_value" in encoder_dict:
                setattr(enc, "encoded_missing_value", encoded_missing)
            if "n_features_in" in encoder_dict:
                enc.n_features_in_ = int(encoder_dict["n_features_in"])
            if "feature_names_in" in encoder_dict:
                enc.feature_names_in_ = np.array(
                    encoder_dict["feature_names_in"], dtype=object
                )
            if "_missing_indices" in encoder_dict:
                mi = encoder_dict["_missing_indices"]
                setattr(
                    enc,
                    "_missing_indices",
                    (
                        {int(k): v for k, v in mi.items()}
                        if isinstance(mi, dict)
                        else mi
                    ),
                )
            if "_infrequent_enabled" in encoder_dict:
                setattr(
                    enc,
                    "_infrequent_enabled",
                    encoder_dict["_infrequent_enabled"],
                )
            label_encoders[variable] = enc
            continue

        raise ValueError(
            f"Unknown encoder type in artifacts JSON for '{variable}': {encoder_type}"
        )

    deserialized["transforms"] = transforms
    deserialized["label_encoders"] = label_encoders
    return deserialized


def _load_artifacts(artifacts_path):
    file_type, raw = load_and_sniff_artifacts(artifacts_path)
    if file_type == "json":
        return _deserialize_json_artifacts(raw)
    return raw


def reconstruct_model(safetensors_path, config_path, artifacts_path, device="cpu"):
    """
    Reconstruct a full Flexynesis model from:
      - safetensors_path : .safetensors weights file
      - config_path      : final_model_config.json
      - artifacts_path   : .artifacts.joblib/json
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

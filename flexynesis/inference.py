# flexynesis/inference.py
# NOTE: Drop-in file. Adds minimal inference on top of the existing receipt stub.
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple

import joblib
import torch

# Optional deps used only for I/O & tabulation
import pandas as pd
import numpy as np


@dataclass
class InferenceArtifacts:
    """
    Container describing what a trained Flexynesis run expects at inference time.

    Attributes
    ----------
    schema_version : int
        Schema version for forward/backward compatibility.
    data_types : list[str]
        Omics layers used during training (e.g., ["gex", "cnv"]).
    target_variables : list[str]
        Targets used during training (may be empty).
    feature_lists : dict[str, list[str]]
        Per-layer feature names observed after preprocessing.
    transforms : dict[str, Any]
        Fitted scalers/encoders per layer (optional).
    label_encoders : dict[str, Any]
        Fitted encoders for categorical clinical variables (optional).
    join_key : str
        Column in clin.csv used to join tables.
    """
    schema_version: int = 1
    data_types: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    feature_lists: Dict[str, List[str]] = field(default_factory=dict)
    transforms: Dict[str, Any] = field(default_factory=dict)
    label_encoders: Dict[str, Any] = field(default_factory=dict)
    join_key: str = "JoinKey"

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of artifacts."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save artifacts to ``path`` using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.to_dict(), path)

    @staticmethod
    def load(path: str) -> "InferenceArtifacts":
        """Load artifacts from ``path`` using joblib."""
        d = joblib.load(path)
        return InferenceArtifacts(**d)


def _log(msg: str) -> None:
    print(f"[flexynesis] {msg}")


# -------------------------
# Minimal inference helpers
# -------------------------
def _load_layer_csv(folder: str, layer: str) -> pd.DataFrame:
    """
    Load a layer matrix from `<folder>/<layer>.csv`.

    Expected shape
    --------------
    rows = samples, columns = features
    First column may be the sample id; otherwise row index should contain sample ids.
    """
    path = os.path.join(folder, f"{layer}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"expected layer file not found: {path}")
    df = pd.read_csv(path)
    if df.shape[1] >= 2 and df.columns[0].lower() in {"sample", "sample_id", "id"}:
        df = df.set_index(df.columns[0])
    else:
        # if no explicit id column, assume index already has ids
        if df.index.name is None:
            # fall back to using the first column as index if it looks non-numeric
            df = df.set_index(df.columns[0])
    return df


def _load_clin_csv(folder: str) -> pd.DataFrame:
    """Load `<folder>/clin.csv` (rows=samples)."""
    path = os.path.join(folder, "clin.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"expected clin file not found: {path}")
    return pd.read_csv(path)


def _strict_check_and_order(
    frames: Dict[str, pd.DataFrame],
    feature_lists: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Ensure each layer has an **exact** feature match and reorder columns accordingly.

    Raises
    ------
    ValueError
        If any feature is missing or extra compared to training artifacts.
    """
    ordered: Dict[str, pd.DataFrame] = {}
    for layer, required in feature_lists.items():
        if layer not in frames:
            raise ValueError(f"test data missing required layer: {layer}")
        df = frames[layer]
        cols = list(df.columns)
        req_set, col_set = set(required), set(cols)
        missing = sorted(req_set - col_set)
        extra = sorted(col_set - req_set)
        if missing or extra:
            raise ValueError(
                f"[minimal inference] feature mismatch in layer '{layer}': "
                f"missing={len(missing)} extra={len(extra)}. "
                f"Minimal path requires an exact match."
            )
        ordered[layer] = df.loc[:, required]  # reorder to training order
    return ordered


def _load_model_checkpoint(pretrained_model_path: str, device: str) -> torch.nn.Module:
    """
    Load a pretrained model.

    Supports
    --------
    - TorchScript via torch.jit.load
    - Pickled nn.Module / LightningModule via torch.load

    Raises
    ------
    RuntimeError
        If a raw state_dict is provided without the model class to instantiate.
    """
    # Try TorchScript first
    try:
        model = torch.jit.load(pretrained_model_path, map_location=device)
        model.eval()
        _log("loaded TorchScript model.")
        return model
    except Exception:
        pass

    # Fallback: pickled module
    obj = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        _log("loaded pickled nn.Module model.")
        return obj

    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        raise RuntimeError(
            "Provided checkpoint looks like a raw state_dict. "
            "Minimal inference cannot build a model class from a state_dict. "
            "Please export a TorchScript module or a full pickled nn.Module."
        )

    raise RuntimeError("Unsupported checkpoint format for minimal inference.")


def _make_tensor_batches(
    ordered_frames: Dict[str, pd.DataFrame],
    device: str,
    batch_size: int = 128,
) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """
    Convert ordered per-layer DataFrames into dict-tensor batches and preserve sample order.

    Returns
    -------
    batches : list[dict[str, torch.Tensor]]
        Each batch is a dict[layer] -> float32 tensor of shape (B, F_layer).
    sample_ids : list[str]
        Sample ids in the same order the batches will be iterated (to reconstruct outputs).
    """
    sample_index = None
    for df in ordered_frames.values():
        idx = list(df.index)
        sample_index = idx if sample_index is None else [s for s in sample_index if s in set(idx)]
    if sample_index is None:
        raise ValueError("No samples found in the provided matrices.")
    aligned = {k: v.loc[sample_index] for k, v in ordered_frames.items()}

    n = len(sample_index)
    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch: Dict[str, torch.Tensor] = {}
        for layer, df in aligned.items():
            x = torch.tensor(df.iloc[start:end].values, dtype=torch.float32, device=device)
            batch[layer] = x
        batches.append(batch)
    return batches, sample_index


def _model_predict_proba(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Try to obtain class probabilities from the model for a single batch.

    Strategy
    --------
    1) Try model(batch) with dict input (common in multi-omics models).
    2) Try model(**batch) (if forward takes named tensors).
    3) If logits returned, apply softmax; if already probs, pass through.

    Raises
    ------
    RuntimeError
        If neither calling convention works.
    """
    with torch.no_grad():
        try:
            out = model(batch)
        except TypeError:
            try:
                out = model(**batch)
            except Exception as e:
                raise RuntimeError(
                    "Cannot call model with provided batch. "
                    "Minimal inference expects `forward(dict)` or `forward(**layers)`."
                ) from e

        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise RuntimeError("Model output is not a tensor in minimal inference path.")

        if out.ndim == 2:
            probs = torch.softmax(out, dim=1)
        else:
            raise RuntimeError("Model output must be 2D (batch, num_classes) for minimal inference.")
        return probs


def _write_predictions_csv(
    out_path: str,
    sample_ids: List[str],
    probs: np.ndarray,
) -> None:
    """Write `<prefix>.predictions.csv` with sample_id, predicted_label, and per-class probabilities."""
    n_classes = probs.shape[1]
    pred = probs.argmax(axis=1)
    cols = ["sample_id", "predicted_label"] + [f"prob_class{i}" for i in range(n_classes)]
    arr = np.column_stack([np.array(sample_ids, dtype=object), pred, probs])
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(out_path, index=False)


def run_inference(
    *,
    model: torch.nn.Module | dict,
    artifacts_path: str,
    data_path_test: str,
    outdir: str,
    prefix: str,
) -> None:
    """
    Run Flexynesis in inference mode.

    Current behavior
    ----------------
    1) Load artifacts and emit a receipt (existing behavior).
    2) **Minimal inference (happy path):**
       - Load test matrices `<layer>.csv` and `clin.csv`.
       - Enforce exact feature match and order.
       - Load a TorchScript or pickled nn.Module checkpoint.
       - Run a forward pass and write `<prefix>.predictions.csv`.
       - Update the receipt with counts and the predictions path.

    Notes
    -----
    This path does not handle missing features/modalities and does not apply
    transforms/encoders. It is intentionally strict as per the minimal spec.
    """
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"artifacts path does not exist: {artifacts_path}")
    if not os.path.isdir(data_path_test):
        raise FileNotFoundError(f"--data_path_test must be a folder: {data_path_test}")

    os.makedirs(outdir, exist_ok=True)

    # 1) Load artifacts and write the base receipt
    arts = InferenceArtifacts.load(artifacts_path)

    _log("Inference mode reached ✅")
    _log(f"  artifacts: {artifacts_path}")
    _log(f"  data_path_test: {data_path_test}")
    _log(f"  outdir/prefix: {outdir} / {prefix}")
    _log(f"  layers: {arts.data_types}")
    _log(f"  targets: {arts.target_variables if arts.target_variables else '(none)'}")
    _log(f"  join_key: {arts.join_key}")

    receipt_path = os.path.join(outdir, f"{prefix}.inference_receipt.json")
    receipt = {
        "status": "ok",
        "mode": "inference",
        "artifacts_path": artifacts_path,
        "data_path_test": data_path_test,
        "outdir": outdir,
        "prefix": prefix,
        "layers": arts.data_types,
        "targets": arts.target_variables,
        "join_key": arts.join_key,
    }

    # 2) Minimal inference (happy path only)
    clin = _load_clin_csv(data_path_test)
    if arts.join_key not in clin.columns:
        raise ValueError(
            f"join_key '{arts.join_key}' not found in clin.csv columns: {list(clin.columns)[:8]}..."
        )
    frames: Dict[str, pd.DataFrame] = {}
    for layer in arts.data_types:
        frames[layer] = _load_layer_csv(data_path_test, layer)

    ordered = _strict_check_and_order(frames, arts.feature_lists)

    if set(clin[arts.join_key]) & set(next(iter(ordered.values())).index):
        sample_order = clin[arts.join_key].tolist()
        ordered = {k: v.reindex(sample_order).dropna(axis=0, how="any") for k, v in ordered.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = _load_model_checkpoint(
        pretrained_model_path=str(getattr(model, "checkpoint_path", None) or model),
        device=device,
    )

    batches, sample_ids = _make_tensor_batches(ordered, device=device, batch_size=256)
    probs_all: List[np.ndarray] = []
    for b in batches:
        probs = _model_predict_proba(mdl, b)
        probs_all.append(probs.detach().cpu().numpy())
    probs_all_np = np.vstack(probs_all)

    pred_path = os.path.join(outdir, f"{prefix}.predictions.csv")
    _write_predictions_csv(pred_path, sample_ids, probs_all_np)
    _log(f"wrote predictions: {pred_path}")

    receipt.update(
        {
            "n_samples": len(sample_ids),
            "n_features_per_layer": {k: len(v.columns) for k, v in ordered.items()},
            "predictions_csv": pred_path,
        }
    )
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    _log(f"updated receipt: {receipt_path}")

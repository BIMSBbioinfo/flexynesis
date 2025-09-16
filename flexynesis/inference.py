"""Inference utilities and artifacts schema for Flexynesis."""
# flexynesis/inference.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any

import joblib
import torch


@dataclass
class InferenceArtifacts:
    """
    Container describing what a trained Flexynesis run expects at inference time.

    Attributes include schema version, layers, targets, per-layer feature lists, fitted transforms/encoders, and the clinical join key.
    """

    schema_version: int = 1
    data_types: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    feature_lists: Dict[str, List[str]] = field(default_factory=dict)
    transforms: Dict[str, Any] = field(default_factory=dict)
    label_encoders: Dict[str, Any] = field(default_factory=dict)
    join_key: str = "JoinKey"

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation of the artifacts."""
        d = asdict(self)
        return d

    def save(self, path: str) -> None:
        """Serialize and save the artifacts to ``path`` using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.to_dict(), path)

    @staticmethod
    def load(path: str) -> "InferenceArtifacts":
        """Load artifacts from ``path`` using joblib and return an instance."""
        d = joblib.load(path)
        return InferenceArtifacts(**d)


def _log(msg: str) -> None:
    print(f"[flexynesis] {msg}")


def run_inference(
    *,
    model: torch.nn.Module | dict,
    artifacts_path: str,
    data_path_test: str,
    outdir: str,
    prefix: str,
) -> None:
    """
    Inference entry-point used by the CLI when --pretrained_model + --artifacts + --data_path_test are provided.

    Current behavior (MVP):
      - Load InferenceArtifacts
      - Print a clear summary of what will be expected/used
      - Early-exit “OK” so CI and users can validate the plumbing end-to-end

    NOTE: This is intentionally minimal to get the wiring in place. A follow-up patch
    will harmonize features, load the test matrices, and produce predictions.
    """
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"artifacts path does not exist: {artifacts_path}")
    if not os.path.isdir(data_path_test):
        raise FileNotFoundError(f"--data_path_test must be a folder: {data_path_test}")

    os.makedirs(outdir, exist_ok=True)

    # Load artifacts
    arts = InferenceArtifacts.load(artifacts_path)

    # Summarize what we have
    _log("Inference mode stub reached ✅")
    _log(f"  artifacts: {artifacts_path}")
    _log(f"  data_path_test: {data_path_test}")
    _log(f"  outdir/prefix: {outdir} / {prefix}")
    _log(f"  layers: {arts.data_types}")
    _log(f"  targets: {arts.target_variables if arts.target_variables else '(none)'}")
    _log(f"  join_key: {arts.join_key}")

    # Touch a tiny JSON “receipt” so users get an output file in CI
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
    with open(os.path.join(outdir, f"{prefix}.inference_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)

    # IMPORTANT:
    # The actual prediction path (reading test matrices, harmonizing features with arts.feature_lists,
    # and writing predictions/embeddings) will be implemented next in this branch.
    # Keeping this lean ensures the CLI contract and CI pass first.

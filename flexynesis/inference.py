from __future__ import annotations
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

SCHEMA_VERSION = 1

@dataclass
class InferenceArtifacts:
    schema_version: int
    data_types: List[str]
    target_variables: List[str]
    feature_lists: Dict[str, List[str]]
    transforms: Dict[str, Any]
    label_encoders: Dict[str, Any]
    join_key: str = "JoinKey"

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": self.schema_version,
            "data_types": self.data_types,
            "target_variables": self.target_variables,
            "feature_lists": self.feature_lists,
            "join_key": self.join_key,
        }
        joblib.dump(
            {"meta": meta, "transforms": self.transforms, "label_encoders": self.label_encoders},
            path
        )

    @staticmethod
    def load(path) -> "InferenceArtifacts":
        blob = joblib.load(path)
        meta = blob["meta"]
        return InferenceArtifacts(
            schema_version=meta["schema_version"],
            data_types=list(meta["data_types"]),
            target_variables=list(meta["target_variables"]),
            feature_lists={k: list(v) for k, v in meta["feature_lists"].items()},
            transforms=blob["transforms"],
            label_encoders=blob["label_encoders"],
            join_key=meta.get("join_key", "JoinKey"),
        )

def run_inference(*args, **kwargs):
    """Placeholder: full implementation will follow in this branch."""
    raise NotImplementedError("Inference mode WIP. See PR discussion in #105.")

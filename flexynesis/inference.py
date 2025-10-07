"""Inference utilities and artifacts schema for Flexynesis."""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import joblib
import torch
import pandas as pd

@dataclass
class InferenceArtifacts:
    """Container for trained model artifacts."""
    schema_version: int = 1
    data_types: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    feature_lists: Dict[str, List[str]] = field(default_factory=dict)
    transforms: Dict[str, Any] = field(default_factory=dict)
    label_encoders: Dict[str, Any] = field(default_factory=dict)
    join_key: str = "JoinKey"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self.to_dict(), path)
    
    @staticmethod
    def load(path: str) -> "InferenceArtifacts":
        d = joblib.load(path)
        return InferenceArtifacts(**d)

def run_inference(
    *,
    model: torch.nn.Module,
    artifacts_path: str,
    data_path_test: str,
    outdir: str,
    prefix: str,
) -> None:
    """Complete inference implementation using DataImporterInference."""
    from .data import DataImporterInference
    
    print("[INFO] Starting inference...")
    print(f"[INFO]   Model: {type(model).__name__}")
    print(f"[INFO]   Artifacts: {artifacts_path}")
    print(f"[INFO]   Test data: {data_path_test}")
    
    # Load test data
    importer = DataImporterInference(
        test_data_path=data_path_test,
        artifacts_path=artifacts_path,
        verbose=True
    )
    
    test_dataset = importer.import_data()
    
    # Run predictions
    print("[INFO] Running predictions...")
    model.eval()
    with torch.no_grad():
        import numpy as np
        
        # Get class predictions (argmax of probabilities)
        raw_preds = model.predict(test_dataset)
        
        if isinstance(raw_preds, dict):
            predictions = {}
            for task, probs in raw_preds.items():
                # Convert to numpy if tensor
                if hasattr(probs, 'cpu'):
                    probs = probs.cpu().numpy()
                # For classification: get argmax (predicted class)
                if len(probs.shape) > 1 and probs.shape[1] > 1:
                    predictions[task] = np.argmax(probs, axis=1)
                else:
                    predictions[task] = probs
        else:
            # Convert to numpy if tensor
            if hasattr(raw_preds, 'cpu'):
                raw_preds = raw_preds.cpu().numpy()
            # For classification: get argmax
            if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
                predictions = np.argmax(raw_preds, axis=1)
            else:
                predictions = raw_preds
    
    # Save predictions
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, f"{prefix}_predictions.csv")
    
    if isinstance(predictions, dict):
        for task, preds in predictions.items():
            task_file = os.path.join(outdir, f"{prefix}_{task}_predictions.csv")
            preds_array = preds.cpu().numpy() if hasattr(preds, 'cpu') else preds
            # Flatten if multi-dimensional
            if len(preds_array.shape) > 1:
                preds_array = preds_array.ravel()
            df = pd.DataFrame({
                'sample': test_dataset.samples,
                'prediction': preds_array
            })
            df.to_csv(task_file, index=False)
            print(f"[INFO] Saved {task} predictions to {task_file}")
    else:
        preds_array = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
        # Flatten if multi-dimensional
        if len(preds_array.shape) > 1:
            preds_array = preds_array.ravel()
        df = pd.DataFrame({
            'sample': test_dataset.samples,
            'prediction': preds_array
        })
        df.to_csv(output_file, index=False)
        print(f"[INFO] Saved predictions to {output_file}")
    
    print("[INFO] Inference complete!")

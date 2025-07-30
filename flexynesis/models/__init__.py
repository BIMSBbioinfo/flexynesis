"""Model implementations available in :mod:`flexynesis`.

The original module eagerly imported all model classes which required
dependencies like ``torch`` at package import time.  To keep ``import
flexynesis`` lightweight we lazily load individual models when they are first
accessed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "DirectPred",
    "supervised_vae",
    "MultiTripletNetwork",
    "CrossModalPred",
    "GNN",
]

_module_map = {
    "DirectPred": "direct_pred",
    "supervised_vae": "supervised_vae",
    "MultiTripletNetwork": "triplet_encoder",
    "CrossModalPred": "crossmodal_pred",
    "GNN": "gnn_early",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny wrapper
    if name in _module_map:
        module = import_module(f".{_module_map[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - small utility
    return sorted(list(__all__))


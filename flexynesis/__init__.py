"""Flexynesis public API.

This package exposes a large collection of deep learning utilities.  Importing
all submodules eagerly pulls in heavy dependencies such as ``torch`` which
slows down the import of :mod:`flexynesis` and makes the package unusable in
minimal environments (e.g. during testing).

To keep the initial import lightweight we lazily import submodules and forward
attribute access to them on demand.  This preserves the original public API
while avoiding unnecessary imports at package import time.
"""

from importlib import import_module
from types import ModuleType
from typing import Any, Dict

__all__ = [
    "modules",
    "data",
    "main",
    "models",
    "feature_selection",
    "utils",
    "config",
]

_imported: Dict[str, ModuleType] = {}


def _load_module(name: str) -> ModuleType:
    if name not in _imported:
        _imported[name] = import_module(f".{name}", __name__)
    return _imported[name]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name in __all__:
        return _load_module(name)

    for mod_name in __all__:
        module = _load_module(mod_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - small utility
    result = set(__all__)
    for mod_name in __all__:
        module = _imported.get(mod_name)
        if module and hasattr(module, "__all__"):
            result.update(getattr(module, "__all__"))
    return sorted(result)


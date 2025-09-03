# Lazy imports to avoid slow startup
# Only import essential components that are needed for basic functionality

# Import core modules without heavy dependencies
from .config import search_spaces

class LazyModule:
    """Lazy module that only imports when accessed."""
    
    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None
        self._import_error = None
    
    def _import_module(self):
        if self._module is None and self._import_error is None:
            try:
                import importlib
                self._module = importlib.import_module(f'.{self._module_name}', package=__name__)
            except ImportError as e:
                self._import_error = e
                raise ImportError(f"Failed to import {self._module_name} module: {e}. "
                               f"This usually means some dependencies are missing. "
                               f"Try installing required packages or check your environment.")
        elif self._import_error is not None:
            raise self._import_error
        return self._module
    
    def __getattr__(self, name):
        module = self._import_module()
        return getattr(module, name)
    
    def __dir__(self):
        if self._module is None:
            return []
        module = self._import_module()
        return dir(module)
    
    def __repr__(self):
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not yet imported)>"
        else:
            return f"<LazyModule '{self._module_name}' (imported)>"

# Create lazy module proxies - these are NOT imported yet
modules = LazyModule('modules')
data = LazyModule('data')
main = LazyModule('main')
models = LazyModule('models')
feature_selection = LazyModule('feature_selection')
utils = LazyModule('utils')

# Import commonly used classes directly for easy access
# These will be imported lazily when first accessed
def _get_data_importer():
    """Lazy getter for DataImporter class."""
    return data.DataImporter

def _get_models():
    """Lazy getter for model classes."""
    return models

# Export all modules and commonly used classes
__all__ = [
    "search_spaces",
    "modules",
    "data", 
    "main",
    "models",
    "feature_selection",
    "utils",
    "DataImporter"
]


# Lazy imports to avoid slow startup
# Only import essential components that are needed for basic functionality

# Import core modules without heavy dependencies
from .config import search_spaces

# Lazy import function for heavy modules
def _import_heavy_modules():
    """Import heavy modules only when needed."""
    # Import modules individually to avoid syntax errors
    from . import modules
    from . import data
    from . import main
    from . import models
    from . import feature_selection
    from . import utils
    return True

# Export search_spaces for immediate access
__all__ = ["search_spaces", "_import_heavy_modules"]

"""
Supporting lifecycle hooks.

Any modules within the `model` module will be checked by finding a list of hook
functions, and these functions will be executed during specific events.

Available hooks:
    neurdb_on_start(): executed when the AI engine starts
"""

import importlib
import pkgutil
from typing import Callable, List

from neurdbrt.log import logger


_on_start_hooks: List[Callable[[], None]] = []


def _find_child_modules(module_path) -> List[str]:
    """
    Find all child modules within a given module or package.

    Args:
        module_path (str): The name of the module or package (e.g., 'mypackage').

    Returns:
        list: List of submodule names.
    """
    try:
        # Import the module/package
        module = importlib.import_module(module_path)
        # Get the path to the module/package
        module_dir = module.__path__ if hasattr(module, "__path__") else None

        if module_dir:
            # Iterate over all modules in the package
            return [name for _, name, _ in pkgutil.iter_modules(module_dir)]
        else:
            return []  # Not a package, so no child modules

    except ImportError:
        logger.error(f"Module {module_path} not found")
        return []


def register_hooks():
    """
    Register all hooks.
    """
    BASE_MODULE = "neurdbrt.model"
    
    modules = _find_child_modules(BASE_MODULE)
    for module_name in modules:
        try:
            # Import the module
            mod = importlib.import_module(f"{BASE_MODULE}.{module_name}")
        except ImportError:
            logger.error(f"Module {module_name} not found")
            continue

        for hook in dir(mod):
            if hook == "neurdb_on_start":
                _on_start_hooks.append(getattr(mod, hook))
                logger.info(f"Registered hook: {hook} from {module_name}")


def exec_hooks_on_start():
    """
    Execute all registered hooks.
    """
    for hook in _on_start_hooks:
        hook()

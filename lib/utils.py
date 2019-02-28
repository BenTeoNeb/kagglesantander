import importlib
import sys

def force_import(custom_module):
    """
    Force module import even if the module is already loaded
    """

    # Check if custom modules are already loaded
    force_module_reload = False
    if custom_module in sys.modules:
        force_module_reload = True

    # Do the import anyway to get the module object
    module = importlib.import_module(custom_module)

    # Force reload if necessary
    if force_module_reload:
        module = importlib.reload(module)
        print("-> Forced custom module reload", module)

    return module

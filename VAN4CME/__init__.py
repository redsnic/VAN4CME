import pkgutil
import importlib

# --- list all submodules and import them ---
for importer, modname, ispkg in pkgutil.walk_packages(path=__path__,
                                                      prefix='VAN4CME.',
                                                      onerror=lambda x: None):
    importlib.import_module(modname)
__version__ = "1.5"

import importlib
import sys

from .trainings import StableDiffusionTextToImageFineTune, StableDiffusionInpaintingFineTune
from .utils import DreamBoothInpaintingDataset, DreamBoothDataset, PromptDataset

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "utils": [],
    "trainings": []
}

_import_structure['trainings'].extend(["StableDiffusionInpaintingFineTune",
                                       "StableDiffusionTextToImageFineTune"])
_import_structure['utils'].extend([
    "DreamBoothInpaintDataset",
    "DreamBoothTextToImageDataset",
    "PromptDataset"
])


def _load_module(name):
    """Load a module and return it."""
    return importlib.import_module(name)


class LazyLoader:
    def __init__(self, import_structure, extra_objects=None):
        self._import_structure = import_structure
        self._loaded_modules = {}
        self._extra_objects = extra_objects or {}

    def __getattr__(self, item):
        if item in self._extra_objects:
            return self._extra_objects[item]

        for module_name, objects in self._import_structure.items():
            if item in objects:
                if module_name not in self._loaded_modules:
                    self._loaded_modules[module_name] = _load_module(f".{module_name}")
                return getattr(self._loaded_modules[module_name], item)

        raise AttributeError(f"Module {__name__} has no attribute {item}")


sys.modules[__name__] = LazyLoader(_import_structure, extra_objects={"__version__": __version__})

"""
Adding CLI utilities here.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import sys
from importlib import util
from types import ModuleType


def load_package_from_path(pkg_path: str) -> ModuleType:
    """Load a package from a given path. Please refer to https://stackoverflow.com/a/50395128"""
    init_path = os.path.join(pkg_path, "__init__.py")
    assert os.path.exists(init_path)

    name = os.path.basename(pkg_path)
    spec = util.spec_from_file_location(name, init_path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

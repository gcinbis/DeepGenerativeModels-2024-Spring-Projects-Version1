import sys
import os
from importlib import import_module
from easydict import EasyDict

"""
    This function modified from the Genforce library: https://github.com/genforce/genforce
"""
def parse_config(config_file):
    """Parses configuration from python file."""
    assert os.path.isfile(config_file)
    directory = os.path.dirname(config_file)
    filename = os.path.basename(config_file)
    module_name, extension = os.path.splitext(filename)
    assert extension == '.py'
    sys.path.insert(0, directory)
    module = import_module(module_name)
    sys.path.pop(0)
    config = EasyDict()
    for key, value in module.__dict__.items():
        if key.startswith('__'):
            continue
        config[key] = value
    del sys.modules[module_name]
    return config
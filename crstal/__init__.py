"""
Copyright (c) 2014-2016  Shuaib Osman

This module offers an interface to the Cuda Risk Simulation and Trading Analytics Library (CRSTAL)
"""

__author__ = "Shuaib Osman"
__license__ = "Free for non-commercial use"
__all__ = ['version_info', '__version__', '__author__', '__license__', 'ConstructCalculation', 'Parser', 'Workbench']

from .UI import Workbench
from .Config import Parser
from .Calculation import ConstructCalculation
from ._version import version_info, __version__

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'jupyter-crstal',
        'require': 'jupyter-crstal/extension'
    }]

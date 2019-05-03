# -*- coding: utf-8 -*-
import sys
from bohrium_api._bh_api import flush
from .ufuncs import ufunc_dict
from .array_create import *
from .bharray import BhArray, BhBase
from . import random, user_kernel

# Expose ufuncs via their names.
for _name, _ufunc in ufunc_dict.items():
    setattr(sys.modules[__name__], _name, _ufunc)


__all__ = ['BhArray', 'BhBase']
for _name in globals():
    if not _name.startswith("_") and _name not in ['sys', 'ufunc_dict']:
        __all__.append(_name)


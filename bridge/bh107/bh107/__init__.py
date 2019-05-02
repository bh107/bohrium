# -*- coding: utf-8 -*-
"""
============================================
bh107: automatic parallelization <bh107.org>
============================================
"""

import sys
from bohrium_api._bh_api import flush
from .ufuncs import ufunc_dict
from .array_create import *
from .bharray import BhArray, BhBase
from . import random, user_kernel

# Expose ufuncs via their names.
for _name, _ufunc in ufunc_dict.items():
    setattr(sys.modules[__name__], _name, _ufunc)

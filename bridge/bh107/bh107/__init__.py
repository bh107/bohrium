# -*- coding: utf-8 -*-
"""
============================================
bh107: automatic parallelization <bh107.org>
============================================
"""

import sys
from .ufuncs import ufunc_dict
from .array_create import *

# Expose ufuncs via their names.
# Notice, we do not expose the `bhop_dict`
for _name, _ufunc in ufunc_dict.items():
    setattr(sys.modules[__name__], '_name', _ufunc)

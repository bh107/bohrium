# -*- coding: utf-8 -*-
"""
================================
pgas4py: bringing PGAS to Python
================================
"""
from __future__ import absolute_import
from pkg_resources import get_distribution, DistributionNotFound

# Set the package version
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0"  # package is not installed

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In this module we implement "python -m bohrium --info"
"""

import argparse
from . import stack_info
from ._bh_api import sanity_check

parser = argparse.ArgumentParser(description='Check and retrieve info on the Bohrium API installation.')
parser.add_argument(
    '--info',
    action="store_true",
    default=False,
    help='Print Runtime Info'
)
parser.add_argument(
    '--no-check',
    action="store_true",
    default=False,
    help='Skip installation check'
)
args = parser.parse_args()
if args.info:
    print(stack_info.pprint())

if not args.no_check:
    if sanity_check():
        print("Installation check succeeded")
    else:
        print("Installation check failed")
        exit(-1)

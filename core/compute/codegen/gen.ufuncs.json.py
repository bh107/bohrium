#!/usr/bin/env python
import json as marshal
import sys
import os
"""
    Extracts ufunc definitions from numpy and prints them out in json format.
"""

if __name__ == "__main__":

    ufuncs  = {}
    errors  = []

    try:
        sys.path.append( os.getcwd()+ "/../../../bridge/numpy/numpy/core/code_generators/" )
    except Exception as e:
        errors.append("Failed adding numpy codegen path to sys-path; [%s]." % e)

    try:
        from generate_umath import defdict as ufuncs
    except Exception as e:
        errors.append("Could not import ufuncs from numpy; [%s]." % str(e))

    funcs = [(                  # Extract the numpy ufuncs
        ufuncs[f].opcode,
        ufuncs[f].nin,
        ufuncs[f].nout,
        [(td.in_, td.out) for td in ufuncs[f].type_descriptions]
        ) for f in ufuncs
    ]

    if errors:
        print '\n'.join(errors)
    elif funcs:
        print marshal.dumps(funcs, sort_keys=True, indent=4)
    else:
        print "No ufuncs and no errors either... weird."

#!/usr/bin/env python
"""
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
"""
import json
from os.path import join
import _bh

dtype_npy2bh = {} #NumPy to Bohrium enum conversion e.g. NPY_FLOAT32 to BH_FLOAT32
dtype_npy_supported = [] #List of support NumPy data types e.g. float32

with open(join('/home/madsbk/repos/bohrium/core/codegen','types.json'), 'r') as f:
    dtypes = json.loads(f.read())

    for t in dtypes:
        if t == "unknown":
            continue
        npy = t['enum'].replace("BH", "NPY", 1);
        dtype_npy2bh[npy] = t['enum']
        dtype_npy_supported.append(t['numpy'])

    _bh.dtype_set_map(dtype_npy2bh)

elementwise_opcodes = []; #NumPy/Bohrium ufunc e.g. multiply/BH_MULTIPLY (list of dicts)

with open(join('/home/madsbk/repos/bohrium/core/codegen','opcodes.json'), 'r') as f:
    opcodes = json.loads(f.read())
    for op in opcodes:
        if op['elementwise'] and op['opcode'] != "BH_NONE":
            o = {'opcode': op['opcode'],
                 'doc':    op['doc'],
                 'nop':    op['nop'],
                 'npy':    op['opcode'].lower()[3:]}#Removing BH_
            elementwise_opcodes.append(o)

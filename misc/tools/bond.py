#!/usr/bin/env python
#
# This file is part of Bohrium and copyright (c) 2013 the Bohrium team:
# http://cphvb.bitbucket.org
#
# Bohrium is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 
# of the License, or (at your option) any later version.
#
# Bohrium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the 
# GNU Lesser General Public License along with Bohrium. 
#
# If not, see <http://www.gnu.org/licenses/>.
#
#!/usr/bin/env python
import ConfigParser
import subprocess
import itertools
import argparse
import pprint
import glob
import json
import sys
import re
import os

import bhutils

numpy_map = {
    "BH_ADD":    "np.add",
    "BH_SUBTRACT":    "np.subtract",
    "BH_MULTIPLY":    "np.multiply",
    "BH_DIVIDE":    "np.divide",
    "BH_POWER":    "np.power",
    "BH_ABSOLUTE":    "np.absolute",
    "BH_GREATER":    "np.greater",
    "BH_GREATER_EQUAL":    "np.greater_equal",
    "BH_LESS":    "np.less",
    "BH_LESS_EQUAL":    "np.less_equal",
    "BH_EQUAL":    "np.equal",
    "BH_NOT_EQUAL":    "np.not_equal",
    "BH_LOGICAL_AND":    "np.logical_and",
    "BH_LOGICAL_OR":    "np.logical_or",
    "BH_LOGICAL_XOR":    "np.logical_xor",
    "BH_LOGICAL_NOT":    "np.logical_not",
    "BH_MAXIMUM":    "np.maximum",
    "BH_MINIMUM":    "np.minimum",
    "BH_BITWISE_AND":    "np.bitwise_and",
    "BH_BITWISE_OR":    "np.bitwise_or",
    "BH_BITWISE_XOR":    "np.bitwise_xor",
    "BH_INVERT":    "np.invert",
    "BH_LEFT_SHIFT":    "np.left_shift",
    "BH_RIGHT_SHIFT":    "np.right_shift",
    "BH_COS":    "np.cos",
    "BH_SIN":    "np.sin",
    "BH_TAN":    "np.tan",
    "BH_COSH":    "np.cosh",
    "BH_SINH":    "np.sinh",
    "BH_TANH":    "np.tanh",
    "BH_ARCSIN":    "np.arcsin",
    "BH_ARCCOS":    "np.arccos",
    "BH_ARCTAN":    "np.arctan",
    "BH_ARCSINH":    "np.arcsinh",
    "BH_ARCCOSH":    "np.arccosh",
    "BH_ARCTANH":    "np.arctanh",
    "BH_ARCTAN2":    "np.arctan2",
    "BH_EXP":    "np.exp",
    "BH_EXP2":    "np.exp2",
    "BH_EXPM1":    "np.expm1",
    "BH_LOG":    "np.log",
    "BH_LOG2":    "np.log2",
    "BH_LOG10":    "np.log10",
    "BH_LOG1P":    "np.log1p",
    "BH_SQRT":    "np.sqrt",
    "BH_CEIL":    "np.ceil",
    "BH_TRUNC":    "np.trunc",
    "BH_FLOOR":    "np.floor",
    "BH_RINT":    "np.rint",
    "BH_MOD":    "np.mod",
    "BH_ISNAN":    "np.isnan",
    "BH_ISINF":    "np.isinf",

    "BH_ADD_REDUCE":            "np.add.reduce",
    "BH_MULTIPLY_REDUCE":       "np.multiply.reduce",
    "BH_MINIMUM_REDUCE":        "np.minimum.reduce",
    "BH_MAXIMUM_REDUCE":        "np.maximum.reduce",
    "BH_LOGICAL_AND_REDUCE":    "np.logical_and.reduce",
    "BH_BITWISE_AND_REDUCE":    "np.bitwise_and.reduce",
    "BH_LOGICAL_OR_REDUCE":     "np.logical_or.reduce",
    "BH_BITWISE_OR_REDUCE":     "np.bitwise_or.reduce",
    "BH_LOGICAL_XOR_REDUCE":    "np.logical_xor.reduce",
    "BH_BITWISE_XOR_REDUCE":    "np.bitwise_xor.reduce",

    "BH_RANDOM":    "np.random.random",
    "BH_RANGE":     "np.arange",
    "BH_IDENTITY":  "np.identity",
    "BH_DISCARD":   "np.discard",
    "BH_FREE":      "np.free",
    "BH_SYNC":      "np.sync",
    "BH_NONE":      "np.none",

    "BH_USERFUNC":  "np.userfunc",
}

binary  = "%s(%s, %s, %s)"
unary   = "%s(%s, %s)"
gen     = "%s(%s)"

def merge_kernels(config):
    """
    Creates a shared library named 'bh_libsij.so' containing
    all the functions defined in the 'kernel_path'.
    A 'bh_libsij.idx' is also produced containing all the bohrium
    functions in the '.so' file.
    """

    krn_path = config.get('cpu', 'kernel_path')
    obj_path = config.get('cpu', 'object_path')
    idx_path = "%s%s%s" % (obj_path, os.sep, "bh_libsij_aaaaaa.idx")
    lib_path = "%s%s%s" % (obj_path, os.sep, "bh_libsij_aaaaaa.so")

    if not os.path.exists(krn_path):
        return (None, "kernel_path(%s) does not exist." % krn_path)

    if not os.path.exists(obj_path):
        return (None, "obj_path(%s) does not exist." % obj_path)

    cmd = [c for c in config.get('cpu',
                                 'compiler_cmd').replace('"','').split(' ')
            if c] + [lib_path]

    symbols = []                                # Find the source-files
    sources = []
    files   = []
    for fn in glob.glob("%s%sBH_*.c" % (krn_path, os.sep)):
        m = re.match('.*(BH_.*)_([a-z]{6}).c', fn)
        if m:
            symbol, instance = m.groups()
            if symbol not in symbols:           # Ignore duplicates
                sources.append(open(fn, 'r').read())
                symbols.append(symbol)
                files.append(fn)
    
    source = "\n".join(sources)                 # Compile them
    p = subprocess.Popen(
        cmd,
        stdin   = subprocess.PIPE,
        stdout  = subprocess.PIPE
    )
    out, err = p.communicate(input=source)

    with open(idx_path, 'w+') as fd:            # Create the index-file
        symbols.sort()
        fd.write("\n".join(symbols))

    return (out, err)

def genesis(config, opcodes, types):
    """
    Generate c-source-code for all bytecodes by
    running Bohrium-cpu engine in dump-source mode and
    call all possible functions based on the bytecode definition.
    """

    # Load Bohrium with source-dumping enabled.
    os.environ['BH_VE_CPU_DUMPSRC'] = "1"
    import bohrium as np

    # Grab the bytecode definition
    typemap = dict([(t['enum'], t['numpy']) for t in types
                    if 'UNKNOWN' not in t['c']])

    exclude_type    = ['BH_UNKNOWN']
    exclude_opc     = [
        'BH_RANDOM', 'BH_RANGE', 'BH_IDENTITY',
        'BH_LOGICAL_XOR_REDUCE',  
        'BH_BITWISE_XOR_REDUCE'
    ]  \
    + [opcode['opcode'] for opcode in opcodes if opcode['system_opcode']]

    dimensions = [1,2,3,4]

    operands = {}                                       # Create operands
    for t in (t for t in types if t['enum'] not in exclude_type):
        tn = t['enum']
        if tn not in operands:
            operands[tn] = {}

        for ndim in [1, 2, 3, 4]:                       # Of different dimenions
            if ndim not in operands[tn]:
                operands[tn][ndim] = {}
            for op in [0,1,2]:
                operands[tn][ndim][op] = np.ones(
                    [3]*ndim,
                    dtype = typemap[tn]
                )
    
    # Call all element-wise opcodes
    for opcode in (opcode for opcode in opcodes
                    if opcode['opcode'] not in exclude_opc):
        nop  = opcode['nop']
        func = eval(numpy_map[opcode['opcode']])    # Get a function-pointer

        for typesig in opcode['types']:

            for dim in dimensions:

                if 'REDUCE' in opcode['opcode']:
                    op_setup = [
                        operands[typesig[1]][dim][1]
                    ]
                else:
                    if nop == 3:
                        op_setup = [
                            operands[typesig[1]][dim][1],
                            operands[typesig[2]][dim][2],
                            operands[typesig[0]][dim][0]
                        ]
                    elif nop == 2:
                        op_setup = [
                            operands[typesig[1]][dim][1],
                            operands[typesig[0]][dim][0]
                        ]
                    elif nop == 1:
                        op_setup = [
                            operands[typesig[0]][dim][0]
                        ]
                    else:
                        raise Exception("Unsupported number of operands.")

                a = np.sum(func(*op_setup))         # Call it
                a == 1

    # Call random
    for ndim in dimensions:
        for t in [np.float32, np.float64]:
            a = np.sum(np.random.random(tuple([3]*ndim), dtype = np.float32,
                                    bohrium=True))
            a == 1

    # Call range generator
    for typesigs in (opcode['types'] for opcode in opcodes
                     if 'BH_RANGE' in opcode['opcode']):
        for typesig in typesigs:
            tn = typemap[typesig[0]]
            a = np.sum(np.arange(1,10, bohrium=True, dtype=tn))
            a == 1

    # Call identity
    for typesigs in (opcode['types'] for opcode in opcodes
                     if 'BH_IDENTITY' in opcode['opcode']):
        for typesig in typesigs:
            otype = typesig[0]
            rtype = typesig[1]

            operands[otype][1][1][:] = operands[rtype][1][1][:]
            a = np.sum(operands[otype][1][1])
            a == 1
    
    return (None, None)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Compile bh_libsij.so')
    p.add_argument(
        '--config',
        help='Path to Bohrium config-file.'
    )
    p.add_argument(
        '--genesis',
        help='Run the bytecode genesis-program.',
        action='store_true'
    )
    p.add_argument(
        '--merge_kernels',
        help='Compile all kernels in kernel-path into a shared library.',
        action='store_true'
    )
    p.add_argument(
        'bohrium',
        help='Path to Bohrium source-code.'
    )
    args = p.parse_args()

    config = bhutils.load_config(args.config)
    opcodes, types = bhutils.load_bytecode(args.bohrium)

    try:
        out, err = (None, None)
        if args.genesis:
            out, err = genesis(config, opcodes, types)
        if args.merge_kernels:
            out, err = merge_kernels(config)
    except Exception as e:
        out = "Check the error message."
        err = str(e)

    if err:
        print "Error: %s" % err
    if out:
        print "Info: %s" % out


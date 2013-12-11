#!/usr/bin/env python
import ConfigParser
import subprocess
import argparse
import json
import os

def load_bytecode(path):
    """
    Load/Read the Bohrium bytecode definition from the Bohrium-sourcecode.

    Raises an exception if 'opcodes.json' and 'types.json' cannot be found or
    are invalid.

    Returns (opcodes, types)
    """
    opcodes = json.load(open(os.sep.join([
        path, 'core', 'codegen', 'opcodes.json'
    ])))
    types   = json.load(open(os.sep.join([
        path, 'core', 'codegen', 'types.json'
    ])))

    return (opcodes, types)

def load_config(path=None):
    """
    Load/Read the Bohrium config file and return it as a ConfigParser object.
    If no path is given the following paths are searched::
        
        /etc/bohrium/config.ini
        ${HOME}/.bohrium/config.ini
        ${CWD}/config.ini

    Raises an exception if config-file cannot be found or is invalid.

    Returns config as a ConfigParser object.
    """

    if path and not os.path.exists(path):   # Check the provided path
        raise e("Provided path to config-file [%s] does not exist" % path)

    if not path:                            # Try to search for it
        potential_path = os.sep.join(['etc','bohrium','config.ini'])
        if os.path.exists(potential_path):
            path = potential_path

        potential_path = os.sep.join([os.path.expanduser("~"), '.bohrium',
                                      'config.ini'])
        if os.path.exists(potential_path):
            path = potential_path

        potential_path = os.environ["BH_CONFIG"] if "BH_CONFIG" in os.environ else ""
        if os.path.exists(potential_path):
            path = potential_path

    if not path:                            # If none are found raise exception
        raise e("No config-file provided or found.")

    p = ConfigParser.ConfigParser()         # Try and parse it
    p.read(path)

    return p

def genesis(config, opcodes, types):
    """
    Generate c-source-code for all bytecodes by
    running Bohrium-cpu engine in dump-source mode and
    call all possible functions based on the bytecode definition.
    """

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

    # Load Bohrium with source-dumping enabled.
    os.environ['BH_VE_CPU_JIT_ENABLED']     = "1"
    os.environ['BH_VE_CPU_JIT_PRELOAD']     = "1"
    os.environ['BH_VE_CPU_JIT_OPTIMIZE']    = "0"
    os.environ['BH_VE_CPU_JIT_FUSION']      = "0"
    os.environ['BH_VE_CPU_JIT_DUMPSRC']     = "1"
    import bohrium as np

    # Grab the bytecode definition
    typemap = dict([(t['enum'], t['numpy']) for t in types
                    if 'UNKNOWN' not in t['c']])

    exclude_type    = ['BH_UNKNOWN']
    exclude_opc     = ['BH_RANDOM', 'BH_RANGE', 'BH_IDENTITY']  + \
    [opcode['opcode'] for opcode in opcodes if opcode['system_opcode']]

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
                operands[tn][ndim][op] = {
                    'c': np.ones( [3]*ndim, dtype = typemap[tn] ),
                    's': np.ones( [3]*ndim, dtype = typemap[tn] ),
                    'k': 3
                }
    
    # Call all element-wise opcodes
    for opcode in (opcode for opcode in opcodes
                    if opcode['opcode'] not in exclude_opc):
        nop  = opcode['nop']
        func = eval(numpy_map[opcode['opcode']])    # Get a function-pointer

        for typesig in opcode['types']:

            for dim in dimensions:

                if 'REDUCE' in opcode['opcode']:
                    op_setup = [
                        operands[typesig[1]][dim][1]['c']
                    ]
                else:
                    if nop == 3:
                        op_setup = [
                            operands[typesig[1]][dim][1]['c'],
                            operands[typesig[2]][dim][2]['c'],
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                        op_setup = [
                            operands[typesig[1]][dim][1]['c'],
                            operands[typesig[2]][dim][2]['k'],
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                        op_setup = [
                            operands[typesig[1]][dim][1]['k'],
                            operands[typesig[2]][dim][2]['c'],
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                    elif nop == 2:
                        op_setup = [
                            operands[typesig[1]][dim][1]['c'],
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                        op_setup = [
                            operands[typesig[1]][dim][1]['k'],
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                        op_setup = [
                            operands[typesig[1]][dim][1]['k'],
                            operands[typesig[0]][dim][0]['s']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                        op_setup = [
                            operands[typesig[1]][dim][1]['s'],
                            operands[typesig[0]][dim][0]['s']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                    elif nop == 1:
                        op_setup = [
                            operands[typesig[0]][dim][0]['c']
                        ]
                        a = np.sum(func(*op_setup))         # Call it
                        a == 1

                    else:
                        raise Exception("Unsupported number of operands.")

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

            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['c'][:]
            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['s'][:]
            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['k']
            a = np.sum(operands[otype][1][1]['c'])
            a == 1
    
    return (None, None)


from itertools import chain
import subprocess
import itertools
import pprint
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
}

# Ignore types
ignore_types    = ['BH_UNKNOWN', 'BH_R123', 'BH_COMPLEX64', 'BH_COMPLEX128']
exclude_opc     = ['BH_RANDOM', 'BH_RANGE', 'BH_IDENTITY']

def creation(config, opcodes, types):

    # Grab Bohrium/NumPy
    np = bhutils.import_bohrium()

    dimensions = [1,2,3,4]
    # Filter out the unknown type
    types = [t for t in types if t['enum'] not in ignore_types]

    # Filter out system opcodes
    opcodes = [opcode for opcode in opcodes if not opcode['system_opcode']]

    # Get a map from enum to numpy type
    typemap = dict([(t['enum'], t['numpy']) for t in types])

    #
    # Setup operands
    #

    # Operands of every type, layout and dimensions 1-4
    operands = {}                               # Create operands
    for t in types:
        tn = t['enum']
        if tn not in operands:
            operands[tn] = {}

        for ndim in dimensions:                 # Of different dimenions
            if ndim not in operands[tn]:
                operands[tn][ndim] = {}
            for op in [0,1,2]:                  # Create three
                operands[tn][ndim][op] = {      # Of different layout
                    'C': np.ones([3]*ndim,      dtype = typemap[tn] ),
                    'S': np.ones(pow(3,ndim)*2, dtype = typemap[tn])[::2].reshape([3]*ndim), 
                    'K': 3
                }

    return (opcodes, types, operands)

def genesis(bytecodes, types, operands):

    # 1) Grab Bohrium/NumPy
    np = bhutils.import_bohrium()
   
    earth = []

    # 2) First there was the element-wise operations
    for bytecode in (bytecode for bytecode in bytecodes 
                   if bytecode['opcode'] not in exclude_opc):
        opcode  = bytecode['opcode']
        nop     = bytecode['nop'] if not 'REDUCE' in opcode else "r"

        for typesig in bytecode['types']:
            for layout in bytecode['layout']:
                earth.append([opcode, typesig, layout])

    print "When done the earth should have", len(earth), "species."

    # 3) Execute all opcodes except for RANDOM, RANGE, and IDENTITY
    for opcode, typesig, layout in earth:
        func = eval(numpy_map[opcode])  # Grab the NumPy functions

        # Ignore functions with
        broken = len([t for t in typesig if t in ignore_types])>0
        if broken:
            continue

        for ndim in [1,2,3,4]:          # Setup operands
            if len(typesig) == 3:
                op_setup = [
                    operands[typesig[1]][ndim][1][layout[1]],
                    operands[typesig[2]][ndim][2][layout[2]],
                    operands[typesig[0]][ndim][0][layout[0]]
                ]
            elif len(typesig) == 2:
                op_setup = [
                    operands[typesig[1]][ndim][1][layout[1]],
                    operands[typesig[0]][ndim][0][layout[0]]
                ]
            elif len(typesig) == 1:
                op_setup = [
                    operands[typesig[0]][ndim][0][layout[0]]
                ]
            else:
                print "WTF!"

            #print opcode, func, typesig, layout, op_setup
            try:
                a = func(*op_setup)
                a = np.sum(a)
                a == 1
            except:
                print "Bad things happened when trying to execute", opcode,
                typesig, layout

    """
    # 4) Then range came into the world
    for typesigs in (opcode['types'] for opcode in opcodes
                     if 'BH_RANGE' in opcode['opcode']):
        for typesig in typesigs:
            tn = typemap[typesig[0]]
            a = np.sum(np.arange(1,10, bohrium=True, dtype=tn))
            a == 1
    """

    # 5) Then identity was ensured

    # 6) And uncertainty introduced

    # 7) And the seventh step is to lay back and let the compiler
    #    do the rest of the work

    return (None, None)

"""

    #
    # Call all functions
    #

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
            for output_layout in (output_layouts):
                for input_layout in layouts:
            otype = typesig[0]
            rtype = typesig[1]

            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['c'][:]
            a = np.sum(operands[otype][1][1]['c'])
            a == 1
            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['s'][:]
            a = np.sum(operands[otype][1][1]['c'])
            a == 1
            operands[otype][1][0]['c'][:] = operands[rtype][1][1]['k']
            a = np.sum(operands[otype][1][1]['c'])
            a == 1

            operands[otype][1][0]['s'][:] = operands[rtype][1][1]['c'][:]
            a = np.sum(operands[otype][1][0]['s'])
            a == 1
            operands[otype][1][0]['s'][:] = operands[rtype][1][1]['s'][:]
            a = np.sum(operands[otype][1][0]['s'])
            a == 1
            operands[otype][1][0]['s'][:] = operands[rtype][1][1]['k']
            a = np.sum(operands[otype][1][0]['s'])
            a == 1

    
    # Call reductions, and element-wise opcodes except for identity
    exclude_opc = ['BH_RANDOM', 'BH_RANGE', 'BH_IDENTITY']
    for opcode in (opcode for opcode in opcodes 
                   if opcode['opcode'] not in exclude_opc):
        nop  = opcode['nop']
        func = eval(numpy_map[opcode['opcode']])    # Get a function-pointer

        for typesig in opcode['types']:
            for dim in dimensions:

                if 'REDUCE' in opcode['opcode']:
                    pass
                else:
                    for setup in setups[nop](operands, typesig, dim):
                        a = np.sum(func(*setup))         # Call it
                        a == 1
                

                if 'REDUCE' in opcode['opcode']:
                    op_setup = reductions(operands, typesig, dim)
                else:
                    if nop == 3:
                        for op_setup in binaries(operands, typesig, dim):
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

    return (None, None)
"""


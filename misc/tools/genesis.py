from itertools import chain
import subprocess
import traceback
import itertools
import tempfile
import pprint
import time
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
    "BH_REAL":  "wrap_real",
    "BH_IMAG":   "wrap_imag",

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

    "BH_ADD_ACCUMULATE":        "np.cumsum",
    "BH_MULTIPLY_ACCUMULATE":   "np.cumprod",

    "BH_RANDOM":    "np.random.random",
    "BH_RANGE":     "np.arange",
    "BH_IDENTITY":  "copy_operands",
    "BH_DISCARD":   "np.discard",
    "BH_FREE":      "np.free",
    "BH_SYNC":      "np.sync",
    "BH_NONE":      "np.none",
}

# Ignore types
suppress_types  = ['BH_UNKNOWN', 'BH_R123']
ignore_types    = ['BH_COMPLEX64', 'BH_COMPLEX128']
ignore_types = []

def genesis(bytecodes, types):

    times=[('start', time.time())]  # Why this? Well because it is always fun to know
                                    # how long it took to do everything in the 
                                    # world of bytecode through the ether of Python/NumPy

    # 1) Grab Bohrium/NumPy
    (np, flush) = bhutils.import_bohrium()

    def copy_operands(source, destination):
        destination[:] = source
        destination[:] = 1
        destination[:] = True

    def wrap_real(source, destination):
        destination[:] = np.real(source)

    def wrap_imag(source, destination):
        destination[:] = np.imag(source)

    times.append(('import', time.time()))

    dimensions = [1,2,3,4]

    # Filter out the unknown type
    types = [t for t in types if t['enum'] not in suppress_types]

    # Filter out system opcodes
    bytecodes = [bytecode for bytecode in bytecodes if not bytecode['system_opcode']]

    # Get a map from enum to numpy type
    typemap = dict([(t['enum'], eval("np.{0}".format(t['numpy']))) for t in types])
    type_sh = dict([(t['enum'], t['shorthand']) for t in types])

    #
    # Setup operands of every type, layout and dimensions 1-4
    #
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
                    'K': typemap[tn](3)
                }

    times.append(('setup', time.time()))
    
    earth = []                                  # Flatten bytecode
    for bytecode in (bytecode for bytecode in bytecodes):
        opcode  = bytecode['opcode']

        if "BH_RANDOM" == opcode:               # Hardcoded specialcase for BH_RANDOM
            for layout in bytecode["layout"]:
                earth.append([opcode, ["BH_FLOAT32"], layout])
                earth.append([opcode, ["BH_FLOAT64"], layout])
        else:
            for typesig in bytecode['types']:
                for layout in bytecode['layout']:
                    earth.append([opcode, typesig, layout])

    #
    # Persist the flattened bytecode
    #
    with tempfile.NamedTemporaryFile(delete=False) as fd:
        for opcode, typesig, layout in earth:
            bytecode_str = "%s_%s_%s\n" % (
                opcode, ''.join([type_sh[t] for t in typesig]), ''.join(layout)
            )
            fd.write(bytecode_str)

        print "When done", len(earth), "kernels should be ready in kernel-path."
        print "See the list of function-names in the file [%s]" % fd.name

    times.append(('flatten', time.time()))

    #
    # Execute it 
    #
    for opcode, typesig, layout in earth:
        func = eval(numpy_map[opcode])  # Grab the NumPy functions

        # Ignore functions with signatures containing ignored types
        broken = len([t for t in typesig if t in ignore_types])>0
        if broken:
            continue

        for ndim in [1,2,3,4]:          # Setup operands
            
            if "BH_RANGE" in opcode:    # Specialcases
                op_setup = [
                    1, 10, 1,
                    typemap[typesig[0]],
                    True
                ]
            elif "BH_RANDOM" in opcode:
                op_setup = [
                    [3]*ndim,
                    typemap[typesig[0]],
                    True
                ]
            elif "_REDUCE" in opcode:
                ndim = 2 if ndim == 1 else ndim
                op_setup = [
                    operands[typesig[1]][ndim][1][layout[1]],
                    0,
                    typemap[typesig[0]],
                    operands[typesig[0]][ndim-1][0][layout[0]]
                ]
            elif "_ACCUMULATE" in opcode:
                op_setup = [
                    operands[typesig[1]][ndim][1][layout[1]],
                    0,
                    typemap[typesig[0]],
                    operands[typesig[0]][ndim][1][layout[1]]
                ]               
            else:
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

            try:
                flush()
                func(*op_setup)
                flush()
            except Exception as e:
                print "Error when executing: %s {%s}_%s, err[%s]." % (
                    opcode, ','.join(typesig), ''.join(layout), e
                )
    
    times.append(('execute', time.time()))
    
    bhutils.print_timings(times)
    print "Run 'bohrium --merge_kernels' to create a stand-alone library."

    return (None, None)


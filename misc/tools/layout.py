import itertools
import pprint
import json

import bhutils

def mangle(bytecodes, types):
    """Introduce "layoutmask" to bytecode."""

    # Operand layouts
    array_layouts = ['C', 'S']  # C - Contiguous, S - Strided,
    const_layouts = ['K']       # K - Constant
    
    input_layouts   = array_layouts + const_layouts
    output_layouts  = array_layouts

    layouts = {
        3:      [x for x in itertools.product(output_layouts,
                                              input_layouts,
                                              input_layouts)],
        
        2:      [x for x in itertools.product(output_layouts,
                                              input_layouts)],
        
        1:      [x for x in itertools.product(output_layouts)],

        0:  [],

        "RANDOM":    [x for x in itertools.product(output_layouts,
                                                    const_layouts,
                                                    const_layouts)],

        "REDUCE":       [x for x in itertools.product(output_layouts,
                                                      array_layouts,
                                                      const_layouts)],
    }

    for bytecode in bytecodes:
        nop = bytecode['nop']
        if 'RANDOM' in bytecode['opcode']:
            nop = "RANDOM"
        elif 'REDUCE' in bytecode['opcode']:
            nop = "REDUCE"
        bytecode["layout"] = layouts[nop]

    print bhutils.bytecode_format(bytecodes, 4)

    return (None, None)



#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):

    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix,'..','..','core','codegen','opcodes.json')) as f:
        opcodes = json.loads(f.read())
    with open(join(prefix,'..','cpp','codegen','element_types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types:
            type_map[t[-1]] = {'cpp'     : t[0],
                               'bhc'     : t[1],
                               'name'    : t[2],
                               'bhc_ary' : "bhc_ndarray_%s_p"%t[2]}

    # Let's generate the header and implementation of all array operations
    impl = ""
    for op in opcodes:
        if op['opcode'] in ["BH_REPEAT", "BH_RANDOM", "BH_NONE"]:#We handle random separately and ignore None
            continue
        impl += "// %s: %s\n"%(op['opcode'][3:], op['doc'])
        impl += "// E.g. %s:\n"%(op['code'])

        # Generate functions that takes no operands
        if len(op['types']) == 0:
            continue

        # Generate a function for each type signature
        for type_sig in op['types']:
            for layout in op['layout']:
                impl += "void %s(" % op['opcode'][3:].lower()
                for i, (symbol, t) in enumerate(zip(layout, type_sig)):
                    if i == 0:
                        impl += "BhArray<%s> &out" % type_map[t]['cpp']
                    else:
                        if symbol == "A":
                            impl += ", const BhArray<%s> &in%d" % (type_map[t]['cpp'], i)
                        else:
                            impl += ", %s in%d"%(type_map[t]['cpp'], i)
                impl += ") {\n"
                impl += "\t\n"
                impl += "}\n"
        impl += "\n\n"

    #Let's add header and footer
    impl = """/* Bohrium CXX Bridge: array operation functions. Auto generated! */

#ifndef __BHXX_ARRAY_OPERATIONS_H
#define __BHXX_ARRAY_OPERATIONS_H

#include <bhxx/multi_array.hpp>
#include <bhxx/runtime.hpp>

namespace bhxx {

%s

} // namespace bhxx

#endif
""" % impl

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Finally, let's write the files
    with open(join(args.output,'array_operations.hpp'), 'w') as f:
        f.write(impl)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Generates the array operation source files for the Bohrium CXX bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()
    main(args)



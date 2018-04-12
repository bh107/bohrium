#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse


def main(args):
    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix, '..', '..', 'core', 'codegen', 'opcodes.json')) as f:
        opcodes = json.loads(f.read())

    with open(join(prefix, '..', '..', 'core', 'codegen', 'types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {
                'cpp'     : t['cpp'],
                'bhc'     : t['bhc'],
                'name'    : t['union'],
                'id'      : t['id'],
                'bhc_ary' : "bhc_ndarray_%s_p" % t['union']
            }

    def signature_hash(type_sig, layout):
        def bin(s):
            return str(s) if s <= 1 else bin(s >> 1) + str(s & 1)
        bitstr = ""
        for type_enum, symbol in zip(type_sig, layout):
            type_bits = bin(type_map[type_enum]['id'])  # Convert type enum to a bit string
            type_bits = ("0" * (4 - len(type_bits))) + type_bits  # Zero pad the bit string
            const_bits = "0" if symbol in ['A', '1D'] else "1"
            bitstr  = const_bits + type_bits + bitstr
        return int(bitstr, 2)  # Convert from binary to integer

    # Let's generate the header and implementation of all array operations
    head = ""; impl = ""

    doc = "// Array operation\n"
    impl += doc; head += doc
    decl = "void bhc_op(bhc_opcode opcode, const bhc_dtype types[], const bhc_bool constants[], void *operands[])"
    head += "%s;\n" % decl
    impl += """%s
{
    switch(opcode) {\n""" % decl
    for op in opcodes:
        # We handle random separately and ignore None
        if op['opcode'] in ["BH_RANDOM", "BH_NONE", "BH_TALLY"]:
            continue
        # Ignoring functions that takes no operands
        if len(op['types']) == 0:
            continue

        impl += "        case %s:\n" % op['opcode'].replace("BH_", "BHC_")
        impl += "            switch(signature_hash(%d, types, constants)) {\n" % op['nop']

        # Generate a case for each type signature and layout
        for type_sig in op['types']:
            for layout in op['layout']:
                assert len(layout) == len(type_sig)
                impl += "                case %d: {\n" % signature_hash(type_sig, layout)
                # Load all operands into a variable
                for i, (symbol, t) in enumerate(zip(layout, type_sig)):
                    impl += " " * 20
                    if symbol in ["A", "1D"]:
                        impl += "{0} op{1} = ({0}) operands[{1}];\n".format(type_map[t]['bhc_ary'], i)
                    else:
                        impl += "const {0} op{1} = *(({0}*) operands[{1}]);\n".format(type_map[t]['bhc'], i)
                # Call the bhc array operation
                impl += " " * 20
                impl += "bhc_%s" % (op['opcode'][3:].lower())
                for symbol, t in zip(layout, type_sig):
                    if symbol in ["A", "1D"]:
                        impl += "_A%s" % type_map[t]['name']
                    else:
                        impl += "_K%s" % type_map[t]['name']
                impl += "("
                for i in range(len(type_sig)):
                    if i > 0:
                        impl += ", "
                    impl += "op%d" % i
                impl += ");\n"
                impl += "                    break;\n"
                impl += "                }\n"
        impl += ' ' * 16 + 'default: fprintf(stderr, "bhc_op(): unknown type signature\\n"); assert(1==2); exit(-1);\n'
        impl += "            }\n"
        impl += "            break;\n"
    impl += """        default: fprintf(stderr, "bhc_op(): unknown opcode\\n"); assert(1==2); exit(-1);
    }
}\n"""


    #Let's add header and footer
    head = """/* Bohrium C Bridge: array operation functions. Auto generated! */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

%s
#ifdef __cplusplus
}
#endif

""" % head
    impl = """/* Bohrium C Bridge: array operation functions enum typed. Auto generated! */

#include <bhxx/bhxx.hpp>
#include "bhc.h"

// Returns the hash of the type signature
uint64_t signature_hash(int nop, const bhc_dtype types[], const bhc_bool constants[]) {
    assert(nop < 10); // The signature must fit 64 bit
    uint64_t ret = 0;
    for(int i=0; i < nop; ++i) {
        uint8_t type_bits = types[i];
        uint8_t constant_bit = constants[i];
        ret |= type_bits << (i*5); // Writes the 4 type bits
        ret |= constant_bit << (i*5+4); // Writes the 1 constant bit
    }
    return ret;
}

%s
""" % impl

    # Finally, let's write the files
    with open(join(args.output, 'bhc_array_operations_enum_typed.h'), 'w') as f:
        f.write(head)
    with open(join(args.output, 'bhc_array_operations_enum_typed.cpp'), 'w') as f:
        f.write(impl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the array operation source files for the Bohrium C bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()
    main(args)

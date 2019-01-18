#!/usr/bin/env python
import json
import os
from os.path import join
import argparse


def main(args):
    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix, '..', '..', 'core', 'codegen', 'opcodes.json')) as f:
        opcodes = json.loads(f.read())
    with open(join(prefix, '..', '..', 'core', 'codegen', 'types.json')) as f:
        types = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {
                'cpp': t['cpp'],
                'bhc': t['bhc'],
                'name': t['union'],
                'bhc_ary': "bhc_ndarray_%s_p" % t['union']
            }

    # Let's generate the header and implementation of all array operations
    head = ""
    impl = ""
    for op in opcodes:
        if op['opcode'] in ["BH_RANDOM"]:
            continue
        # Generate functions that takes no operands
        if len(op['types']) == 0:
            continue

        doc = "// %s: %s\n" % (op['opcode'][3:], op['doc'])
        doc += "// E.g. %s:\n" % (op['code'])
        impl += doc
        head += doc

        # Generate a function for each type signature
        for type_sig in op['types']:
            for layout in op['layout']:
                decl = "void %s(" % op['opcode'][3:].lower()
                for i, (symbol, t) in enumerate(zip(layout, type_sig)):
                    if i == 0:
                        decl += "BhArray<%s> &out" % type_map[t]['cpp']
                    else:
                        if symbol == "A":
                            decl += ", const BhArray<%s> &in%d" % (type_map[t]['cpp'], i)
                        else:
                            decl += ", %s in%d" % (type_map[t]['cpp'], i)
                decl += ")"
                head += "%s;\n" % decl
                impl += decl
                impl += "\n{\n\tRuntime::instance().enqueue(%s, out" % op['opcode']
                for i in range(len(layout) - 1):
                    impl += ", in%d" % (i + 1)
                impl += ");\n}\n"

        # Generate a function that returns its output for each type signature
        for type_sig in op['types']:
            if len(type_sig) > 2:
                for layout in op['layout']:
                    out_cpp_type = type_map[type_sig[0]]['cpp']
                    decl = "BhArray<%s> %s(" % (out_cpp_type, op['opcode'][3:].lower())
                    first_input_array = -1
                    for i, (symbol, t) in list(enumerate(zip(layout, type_sig)))[1:]:
                        if i > 1:
                            decl += ", "
                        if symbol == "A":
                            first_input_array = i
                            decl += "const BhArray<%s> &in%d" % (type_map[t]['cpp'], i)
                        else:
                            decl += "%s in%d" % (type_map[t]['cpp'], i)
                    decl += ")"
                    head += "%s;\n" % decl
                    impl += decl
                    impl += "\n{\n"
                    assert (first_input_array != -1)
                    impl += "\tBhArray<{0}> out = BhArray<{0}>".format(out_cpp_type)
                    impl += "{in%(i)d.shape, in%(i)d.stride, in%(i)d.offset};\n" % {"i": first_input_array}
                    impl += "\tRuntime::instance().enqueue(%s, out" % op['opcode']
                    for i in range(len(layout) - 1):
                        impl += ", in%d" % (i + 1)
                    impl += ");\n"
                    impl += "\treturn out;\n"
                    impl += "}\n"
        # Generate an operator overload for each type signature
        operator = {"BH_ADD": "+", "BH_SUBTRACT": "-", "BH_MULTIPLY": "*", "BH_DIVIDE": "/", "BH_MOD": "%",
                    "BH_BITWISE_AND": "&", "BH_BITWISE_OR": "|", "BH_BITWISE_XOR": "^"}
        if op['opcode'] in operator:
            for type_sig in op['types']:
                for layout in op['layout']:
                    out_cpp_type = type_map[type_sig[0]]['cpp']
                    decl = "BhArray<%s> operator%s(" % (out_cpp_type, operator[op['opcode']])
                    first_input_array = -1
                    for i, (symbol, t) in list(enumerate(zip(layout, type_sig)))[1:]:
                        if i > 1:
                            decl += ", "
                        if symbol == "A":
                            first_input_array = i
                            decl += "const BhArray<%s> &in%d" % (type_map[t]['cpp'], i)
                        else:
                            decl += "%s in%d" % (type_map[t]['cpp'], i)
                    decl += ")"
                    head += "%s;\n" % decl
                    impl += decl
                    impl += "\n{\n"
                    assert (first_input_array != -1)
                    impl += "\tBhArray<{0}> out = BhArray<{0}>".format(out_cpp_type)
                    impl += "{in%(i)d.shape, in%(i)d.stride, in%(i)d.offset};\n" % {"i": first_input_array}
                    impl += "\tRuntime::instance().enqueue(%s, out" % op['opcode']
                    for i in range(len(layout) - 1):
                        impl += ", in%d" % (i + 1)
                    impl += ");\n"
                    impl += "\treturn out;\n"
                    impl += "}\n"
        impl += "\n\n"
        head += "\n\n"

    # Let's handle random
    doc = """
/*Fill out with random data.
  The returned result is a deterministic function of the key and counter,
  i.e. a unique (seed, indexes) tuple will always produce the same result.
  The result is highly sensitive to small changes in the inputs, so that the sequence
  of values produced by simply incrementing the counter (or key) is effectively
  indistinguishable from a sequence of samples of a uniformly distributed random variable.

  random123(out, seed, key) where: 'out' is the array to fill with random data
                                   'seed' is the seed of a random sequence
                                   'key' is the index in the random sequence */
"""
    impl += doc
    head += doc
    decl = "void random123(BhArray<uint64_t> &out, uint64_t seed, uint64_t key)"
    head += "%s;\n" % decl
    impl += "%s\n" % decl
    impl += """
{
    \tRuntime::instance().enqueueRandom(out, seed, key);
}
"""

    # Let's add header and footer
    head = """/* Bohrium CXX Bridge: array operation functions. Auto generated! */
#pragma once

#include <cstdint>
#include <complex>

namespace bhxx {

template<typename T> class BhArray;

%s

} // namespace bhxx

""" % head

    impl = """/* Bohrium C Bridge: array operation functions. Auto generated! */

#include <bhxx/Runtime.hpp>
#include <bhxx/array_operations.hpp>

namespace bhxx {

%s

} // namespace bhxx
    """ % impl

    if not os.path.exists(args.inc_output):
        os.makedirs(args.inc_output)
    if not os.path.exists(args.src_output):
        os.makedirs(args.src_output)

    # Finally, let's write the files
    with open(join(args.inc_output, 'array_operations.hpp'), 'w') as f:
        f.write(head)
    with open(join(args.src_output, 'array_operations.cpp'), 'w') as f:
        f.write(impl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the array operation source files for the Bohrium CXX bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'inc_output',
        help='Path to the header output directory.'
    )
    parser.add_argument(
        'src_output',
        help='Path to the source output directory.'
    )
    args = parser.parse_args()
    main(args)

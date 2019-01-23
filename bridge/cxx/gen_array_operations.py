#!/usr/bin/env python
import json
import os
from os.path import join
import argparse


def write_decl(op, layout, type_sig, type_map, operator, out_as_operand, compound_assignment):
    signature = list(enumerate(zip(layout, type_sig)))
    out_cpp_type = type_map[type_sig[0]]['cpp']

    if operator is None:
        func_name = "%s" % op['opcode'][3:].lower()
    else:
        func_name = "operator%s" % operator
    if out_as_operand:
        decl = "void %s(BhArray<%s> &out" % (func_name, out_cpp_type)
        if len(signature) > 1:
            decl += ", "
    else:
        decl = "BhArray<%s> " % out_cpp_type
        if compound_assignment:
            decl += "&"  # compound assignment such as "+=" returns a reference
        decl += "%s(" % func_name

    for i, (symbol, t) in signature[1:]:
        if symbol == "A":
            if not (i == 1 and compound_assignment):
                decl += "const "
            decl += "BhArray<%s> &in%d" % (type_map[t]['cpp'], i)
        else:
            decl += "%s in%d" % (type_map[t]['cpp'], i)
        if i < len(layout) - 1:
            decl += ", "
    decl += ")"
    return decl


def get_array_inputs(layout, ignore_ops=[]):
    ret = []
    for i, symbol in enumerate(layout):
        if i not in ignore_ops:
            if symbol == "A":
                if i == 0:
                    ret.append("out")
                else:
                    ret.append("in%d" % i)
    return ret


def write_broadcasted_shape(array_inputs):
    ret = "const Shape shape = broadcasted_shape<%d>({" % (len(array_inputs))
    for i, op in enumerate(array_inputs):
        ret += "%s.shape()" % op
        if i < len(array_inputs) - 1:
            ret += ", "
    ret += "});"
    return ret


def write_broadcast_and_enqueue(op, layout, array_inputs):
    ret = ""
    for op_var in array_inputs:
        ret += "\tauto _{0} = broadcast_to({0}, shape);\n".format(op_var)
    ret += "\tRuntime::instance().enqueue(%s, out" % op['opcode']
    for i in range(len(layout) - 1):
        op_var = "in%d" % (i + 1)
        if op_var in array_inputs:
            op_var = "_%s" % op_var
        ret += ", %s" % op_var
    ret += ");\n"
    return ret


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

        ignore_ops = [0]
        if op['opcode'] == "BH_GATHER":
            ignore_ops.append(1)

        # Generate a function for each type signature
        for type_sig in op['types']:
            for layout in op['layout']:
                array_inputs = get_array_inputs(layout, ignore_ops)
                decl = write_decl(op, layout, type_sig, type_map, None, True, False)
                head += "%s;\n" % decl
                impl += decl
                impl += " {\n"
                if op['opcode'] == "BH_IDENTITY" and len(array_inputs) == 1 \
                        and type_map[type_sig[0]]['cpp'] == type_map[type_sig[1]]['cpp']:
                    impl += "\tif (is_same_array(out, in1)) { out.reset(in1); return; }\n"
                if len(array_inputs) > 0:
                    impl += "\t%s\n" % write_broadcasted_shape(array_inputs)
                else:
                    impl += "\tconst Shape &shape = out.shape();\n"

                impl += "\tShape out_shape = shape;\n"
                if "REDUCE" in op['opcode']:
                    impl += "\tif (out_shape.size() == 1) { out_shape = {1}; } else " \
                            "{ out_shape.erase(out_shape.begin() + in2); }\n"

                impl += "\tif (!out.base()) { out.reset(BhArray<%s>{out_shape}); }\n" % type_map[type_sig[0]]['cpp']
                if op['opcode'] not in ['BH_SCATTER', 'BH_COND_SCATTER']:
                    impl += "\tif(out_shape != out.shape()) { throw std::runtime_error(\"Output shape miss match\"); }\n"
                for op_var in get_array_inputs(layout):
                    impl += "\tif(!%s.base()) { throw std::runtime_error(\"Operands not initiated\"); }\n" % op_var
                if len(array_inputs) > 1:
                    for op_var in array_inputs:
                        impl += '\tif(out.base() == {0}.base() && !is_same_array(out, {0})) '.format(op_var)
                        impl += '{ throw std::runtime_error("When output and input uses the same base array, ' \
                                'they must be identical"); }\n'
                impl += write_broadcast_and_enqueue(op, layout, array_inputs)
                impl += "}\n"

        # Generate a function that returns its output for each type signature
        for type_sig in op['types']:
            if len(type_sig) > 1 and op['opcode'] != "BH_IDENTITY":
                for layout in op['layout']:
                    array_inputs = get_array_inputs(layout, ignore_ops)
                    if len(array_inputs) > 0:
                        decl = write_decl(op, layout, type_sig, type_map, None, False, False)
                        head += "%s;\n" % decl
                        impl += decl
                        impl += " {\n"
                        impl += "\tBhArray<%s> out;\n" % type_map[type_sig[0]]['cpp']
                        impl += "\t%s(out" % op['opcode'][3:].lower()
                        for i in range(1, len(type_sig)):
                            impl += ", in%s" % i
                        impl += ");\n"
                        impl += "\treturn out;\n"
                        impl += "}\n"

        # Generate an operator overload for each type signature
        operator = {"BH_ADD": "+", "BH_SUBTRACT": "-", "BH_MULTIPLY": "*", "BH_DIVIDE": "/", "BH_MOD": "%",
                    "BH_BITWISE_AND": "&", "BH_BITWISE_OR": "|", "BH_BITWISE_XOR": "^"}
        if op['opcode'] in operator:
            for type_sig in op['types']:
                for layout in op['layout']:
                    array_inputs = get_array_inputs(layout, ignore_ops)
                    if len(array_inputs) > 0:
                        decl = write_decl(op, layout, type_sig, type_map, operator[op['opcode']], False, False)
                        head += "%s;\n" % decl
                        impl += decl
                        impl += " {\n"
                        impl += "\tBhArray<%s> out;\n" % type_map[type_sig[0]]['cpp']
                        impl += "\t%s(out" % op['opcode'][3:].lower()
                        for i in range(1, len(type_sig)):
                            impl += ", in%s" % i
                        impl += ");\n"
                        impl += "\treturn out;\n"
                        impl += "}\n"

        # Generate += operator overload for each type signature
        if op['opcode'] in operator:
            for type_sig in op['types']:
                for layout in op['layout']:
                    if layout[1] == "A":
                        decl = write_decl(op, layout, type_sig, type_map, "%s=" % operator[op['opcode']], False, True)
                        head += "%s;\n" % decl
                        impl += decl
                        impl += " {\n"
                        impl += "\t%s(in1" % op['opcode'][3:].lower()
                        for i in range(1, len(type_sig)):
                            impl += ", in%s" % i
                        impl += ");\n"
                        impl += "\treturn in1;\n"
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
#include <bhxx/util.hpp>

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

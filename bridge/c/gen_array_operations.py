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
    head = ""; impl = ""
    for op in opcodes:
        if op['opcode'] in ["BH_RANDOM", "BH_NONE"]:#We handle random separately and ignore None
            continue
        doc = "// %s: %s\n"%(op['opcode'][3:], op['doc'])
        doc += "// E.g. %s:\n"%(op['code'])
        impl += doc; head += doc

        # Generate functions that takes no operands
        if len(op['types']) == 0:
            decl = "void bhc_%s()"%(op['opcode'][3:].lower())
            head += "DLLEXPORT %s;\n"%decl
            impl += decl;
            impl += "{\n"
            impl += "\t%s();\n"%op['opcode'].lower()
            impl += "}\n"

        # Generate a function for each type signature
        for type_sig in op['types']:
            for layout in op['layout']:

                for i in xrange(len(layout)):#We need to replace 1D symbols with A
                    if layout[i].endswith("D"):
                        layout[i] = "A"

                decl = "void bhc_%s"%(op['opcode'][3:].lower())
                assert len(layout) == len(type_sig)
                for symbol, t in zip(layout,type_sig):
                    decl += "_%s%s"%(symbol, type_map[t]['name'])
                decl += "(%s out"%type_map[type_sig[0]]['bhc_ary']
                for i, (symbol, t) in enumerate(zip(layout[1:], type_sig[1:])):
                    decl += ", "
                    if symbol == "A":
                        decl += "const %s in%d"%(type_map[t]['bhc_ary'],i+1)
                    else:
                        decl += "%s in%d"%(type_map[t]['bhc'],i+1)
                decl += ")"
                head += "DLLEXPORT %s;\n"%decl
                impl += decl;
                impl += "{\n"
                impl += "\tmulti_array<%(t)s> *o = (multi_array<%(t)s> *) out;\n"%{'t':type_map[type_sig[0]]['cpp']}
                bxx_args = "*o";
                for i, (symbol, t) in enumerate(zip(layout[1:], type_sig[1:])):
                    if symbol in ["A", "D1"]:
                        impl += "\tmulti_array<%(t)s> *i%(i)d = (multi_array<%(t)s> *) in%(i)d;\n"%{'t':type_map[t]['cpp'], 'i':i+1}
                        bxx_args += ", *i%d"%(i+1);
                    else:
                        impl += "\t%s i%d;\n"%(type_map[t]['cpp'], i+1)
                        if t.startswith("BH_COMPLEX"):
                            impl += "\ti%(i)d.real(in%(i)d.real);\n"%{'i':i+1}
                            impl += "\ti%(i)d.imag(in%(i)d.imag);\n"%{'i':i+1}
                        else:
                            impl += "\ti%(i)d = in%(i)d;\n"%{'i':i+1}
                        bxx_args += ", i%d"%(i+1);
                impl += "\t%s(%s);\n"%(op['opcode'].lower(), bxx_args)

                impl += "}\n"
        impl += "\n\n"; head += "\n\n"

    #Let's handle random
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
    impl += doc; head += doc
    decl = "void bhc_random123_Auint64_Kuint64_Kuint64(bhc_ndarray_uint64_p out, uint64_t seed, uint64_t key)"
    head += "DLLEXPORT %s;\n"%decl
    impl += "%s\n"%decl
    impl += """
{
    bh_random<uint64_t>(*((multi_array<uint64_t>*)out), seed, key);
}
"""

    #Let's add header and footer
    head = """/* Bohrium C Bridge: array operation functions. Auto generated! */

#ifndef __BHC_ARRAY_OPERATIONS_H
#define __BHC_ARRAY_OPERATIONS_H

#ifdef _WIN32
#define DLLEXPORT __declspec( dllexport )
#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

%s
#ifdef __cplusplus
}
#endif
#endif
"""%head
    impl = """/* Bohrium C Bridge: array operation functions. Auto generated! */

#include <bohrium.hpp>
#include "bhc.h"

using namespace bxx;

%s
"""%impl

    #Finally, let's write the files
    with open(join(args.output,'bhc_array_operations.h'), 'w') as f:
        f.write(head)
    with open(join(args.output,'bhc_array_operations.cpp'), 'w') as f:
        f.write(impl)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Generates the array operation source files for the Bohrium C bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()
    main(args)

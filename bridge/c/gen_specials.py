#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):

    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix,'..','cpp','codegen','element_types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types:
            type_map[t[-1]] = {'cpp'     : t[0],
                               'bhc'     : t[1],
                               'name'    : t[2],
                               'bhc_ary' : "bhc_ndarray_%s_p"%t[2]}

    # Let's generate the header and implementation of all data types
    head = ""; impl = ""

    doc = "//Create new array\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "%s bhc_new_%s(uint64_t size)"%(t['bhc_ary'], t['name'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s{\n"%decl
        impl += "\tmulti_array<%(t)s> *ret = new multi_array<%(t)s>(size);\n"%{'t':t['cpp']}
        impl += "\tret->setTemp(false);\n\tret->link();\n}\n"

    doc = "//Destroy array\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "void bhc_destroy_%s(%s ary)"%(t['name'], t['bhc_ary'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s{\n"%decl
        impl += "\tdelete ((multi_array<%s>*)ary);\n}\n"%t['cpp']

    doc = "//Create view\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "%s bhc_view_%s("%(t['bhc_ary'], t['name'])
        decl += "const %s src, uint64_t rank, int64_t start, "%t['bhc_ary']
        decl += "const int64_t *shape, const int64_t *stride)"
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s{\n"%decl
        impl += "}\n"

    #Let's add header and footer
    head = """/* Bohrium C Bridge: special functions. Auto generated! */

#ifndef __BHC_SPECIALS_H
#define __BHC_SPECIALS_H

#include <stdint.h>
#include <bh_type.h>

#ifdef __cplusplus
extern "C" {
#endif

%s
#ifdef __cplusplus
}
#endif
#endif
"""%head
    impl = """/* Bohrium C Bridge: special functions. Auto generated! */

#include <bohrium.hpp>
#include "bhc_specials.h"

using namespace bxx;

%s
"""%impl

    print head
    print impl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Generates the special source files for the Bohrium C bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)

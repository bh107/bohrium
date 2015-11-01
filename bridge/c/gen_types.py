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
    head += "//Forward declaration of array types:\n"
    for key, val in type_map.iteritems():
        head += "struct bhc_ndarray_%s;\n"%val['name']
    head += "\n//Pointer shorthands:\n"
    for key, val in type_map.iteritems():
        head += "typedef struct bhc_ndarray_%s* %s;\n"%(val['name'], val['bhc_ary'])

    impl += "//Array types:\n"
    for key, val in type_map.iteritems():
        impl += "struct bhc_ndarray_%s {multi_array<%s> me;};\n"%(val['name'], val['bhc'])

    #Let's add header and footer
    head = """/* Bohrium C Bridge: data types. Auto generated! */

#ifndef __BHC_TYPES_H
#define __BHC_TYPES_H

#include <stdint.h>
#include <bh_type.h>

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
    impl = """/* Bohrium C Bridge: data types. Auto generated! */

#include <bohrium.hpp>
#include "bhc.h"

using namespace bxx;

%s
"""%impl

    #Finally, let's write the files
    with open(join(args.output,'bhc_types.h'), 'w') as f:
        f.write(head)
    with open(join(args.output,'bhc_types.cpp'), 'w') as f:
        f.write(impl)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Generates the type source files for the Bohrium C bridge.',
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

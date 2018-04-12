#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):
    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix, '..', '..', 'core', 'codegen', 'types.json')) as f:
        types    = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {
                'cpp'     : t['cpp'],
                'bhc'     : t['bhc'],
                'name'    : t['union'],
                'id'      : t['id'],
                'bhc_ary' : "bhc_ndarray_%s_p" % t['union']
            }

    # Let's generate the header and implementation of all data types
    head = ""; impl = ""
    head += "// Forward declaration of array types:\n"
    for key, val in type_map.items():
        head += "struct bhc_ndarray_%s;\n" % val['name']
    head += "\n// Pointer shorthands:\n"
    for key, val in type_map.items():
        head += "typedef struct bhc_ndarray_%s* %s;\n" % (val['name'], val['bhc_ary'])

    head += "\n// Type enum:\n"
    head += "typedef enum {\n"
    for key, val in type_map.items():
        head += "    %s = %d, \n" % (key, val['id'])
    head += "} bhc_dtype; // Fits 5-bits\n"

    impl += "// Array types:\n"
    for key, val in type_map.items():
        impl += "struct bhc_ndarray_%s {bhxx::BhArray<%s> me;};\n" % (val['name'], val['cpp'])

    with open(join(prefix, '..', '..', 'core', 'codegen', 'opcodes.json')) as f:
        opcodes = json.loads(f.read())

    head += "\n// Opcodes enum:\n"
    head += "typedef enum {\n"
    for op in opcodes:
        head += "    %s = %s, \n" % (op['opcode'].replace("BH_", "BHC_"), op['id'])
    head += "} bhc_opcode;\n"

    # Let's add header and footer
    head = """/* Bohrium C Bridge: data types. Auto generated! */
#pragma once

#include <stdint.h>

typedef unsigned char bhc_bool;
typedef int8_t        bhc_int8;
typedef int16_t       bhc_int16;
typedef int32_t       bhc_int32;
typedef int64_t       bhc_int64;
typedef uint8_t       bhc_uint8;
typedef uint16_t      bhc_uint16;
typedef uint32_t      bhc_uint32;
typedef uint64_t      bhc_uint64;
typedef float         bhc_float32;
typedef double        bhc_float64;
typedef struct { bhc_float32 real, imag; } bhc_complex64;
typedef struct { bhc_float64 real, imag; } bhc_complex128;
typedef struct { bhc_uint64 start, key; } bhc_r123;

#ifdef __cplusplus
extern "C" {
#endif

%s
#ifdef __cplusplus
}
#endif

""" % head
    impl = """/* Bohrium C Bridge: data types. Auto generated! */

#include <bhxx/bhxx.hpp>
#include "bhc.h"

%s
""" % impl

    # Finally, let's write the files
    with open(join(args.output, 'bhc_types.h'), 'w') as f:
        f.write(head)
    with open(join(args.output, 'bhc_types.cpp'), 'w') as f:
        f.write(impl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the type source files for the Bohrium C bridge.',
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

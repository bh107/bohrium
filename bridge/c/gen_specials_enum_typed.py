#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse


def main(args):
    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix, '..', '..', 'core', 'codegen', 'types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {
                'cpp'     : t['cpp'],
                'bhc'     : t['bhc'],
                'name'    : t['union'],
                'bhc_ary' : "bhc_ndarray_%s_p"%t['union']
            }

    # Let's generate the header and implementation of all data types
    head = ""; impl = ""

    doc = "\n// Create new flat array\n"
    impl += doc; head += doc
    decl = "void *bhc_new(bhc_dtype dtype, uint64_t size)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: return bhc_new_A%s(size);\n" % (key, t['name'])
    impl += """        default: fprintf(stderr, "bhc_new(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Destroy array\n"
    impl += doc; head += doc
    decl = "void bhc_destroy(bhc_dtype dtype, void *ary)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: bhc_destroy_A%s((%s)ary); break;\n" % (key, t['name'], t['bhc_ary'])
    impl += """        default: fprintf(stderr, "bhc_destroy(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Create view of a flat array `src`\n"
    impl += doc; head += doc
    decl = "void *bhc_view(bhc_dtype dtype, void *src, int64_t rank, int64_t start, " \
           "const int64_t *shape, const int64_t *stride)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: return bhc_view_A%s((%s)src, rank, start, shape, stride);\n" \
                % (key, t['name'], t['bhc_ary'])
    impl += """        default: fprintf(stderr, "bhc_view(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Get data pointer from the first VE in the runtime stack\n"
    doc += "//   if 'copy2host', always copy the memory to main memory\n"
    doc += "//   if 'force_alloc', force memory allocation before returning the data pointer\n"
    doc += "//   if 'nullify', set the data pointer to NULL after returning the data pointer\n"
    impl += doc; head += doc
    decl = "void *bhc_data_get(bhc_dtype dtype, const void *ary, bhc_bool copy2host, "
    decl += "bhc_bool force_alloc, bhc_bool nullify)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: return bhc_data_get_A%s((%s)ary, copy2host, force_alloc, nullify);\n" \
                % (key, t['name'], t['bhc_ary'])
    impl += """        default: fprintf(stderr, "bhc_data_get(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Set data pointer in the first VE in the runtime stack\n"
    doc += "// NB: The component will deallocate the memory when encountering a BH_FREE\n"
    doc += "//   if 'host_ptr', the pointer points to the host memory (main memory) as opposed to device memory\n"
    impl += doc; head += doc
    decl = "void bhc_data_set(bhc_dtype dtype, const void *ary, bhc_bool host_ptr, void *data)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: bhc_data_set_A%s((%s)ary, host_ptr, (%s*)data); break;\n" \
                % (key, t['name'], t['bhc_ary'], t['bhc'])
    impl += """        default: fprintf(stderr, "bhc_data_set(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Informs the runtime system to make data synchronized and available after the next flush().\n"
    impl += doc; head += doc
    decl = "void bhc_sync(bhc_dtype dtype, const void *ary)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: bhc_sync_A%s((%s)ary); break;\n" % (key, t['name'], t['bhc_ary'])
    impl += """        default: fprintf(stderr, "bhc_sync(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Slides the view of an array in the given dimensions, by the given strides for each iteration in a loop.\n"
    impl += doc; head += doc
    decl = "void bhc_slide_view(bhc_dtype dtype, const void *ary1, const void *ary2, size_t dim, int slide, int shape)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: bhc_slide_view_A%s_A%s((%s)ary1, (%s)ary2, dim, slide, shape); break;\n" % (key, t['name'], t['name'], t['bhc_ary'], t['bhc_ary'])
    impl += """        default: fprintf(stderr, "bhc_slide_view(): unknown dtype\\n"); exit(-1);
    }
}\n"""

    doc = "\n// Extension Method, returns 0 when the extension exist\n"
    impl += doc; head += doc
    decl = "int bhc_extmethod(bhc_dtype dtype, const char *name, const void *out, const void *in1, const void *in2)"
    head += "DLLEXPORT %s;\n" % decl
    impl += """%s
{
    switch(dtype) {\n""" % decl
    for key, t in type_map.items():
        impl += "        case %s: " % key
        impl += "return bhc_extmethod_A%(name)s_A%(name)s_A%(name)s(name, " \
                "(%(bhc_ary)s)out, (%(bhc_ary)s)in1, (%(bhc_ary)s)in2);\n" % t
    impl += """        default: fprintf(stderr, "bhc_extmethod(): unknown dtype\\n"); exit(-1);
    }
}\n"""


    #Let's add header and footer
    head = """/* Bohrium C Bridge: special functions. Auto generated! */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define DLLEXPORT __declspec( dllexport )
#else
#define DLLEXPORT
#endif

%s
#ifdef __cplusplus
}
#endif

""" % head
    impl = """/* Bohrium C Bridge: special functions enum typed. Auto generated! */

#include <bhxx/bhxx.hpp>
#include "bhc.h"

%s
""" % impl

    #Finally, let's write the files
    with open(join(args.output, 'bhc_specials_enum_typed.h'), 'w') as f:
        f.write(head)
    with open(join(args.output, 'bhc_specials_enum_typed.cpp'), 'w') as f:
        f.write(impl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the special source files for the Bohrium C bridge.',
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

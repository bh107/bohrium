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

    doc = "\n//Flush the Bohrium runtime system\n"
    impl += doc; head += doc
    decl = "void bhc_flush(void)"
    head += "DLLEXPORT %s;\n"%decl
    impl += "%s\n"%decl
    impl += """
{
    Runtime::instance().flush();
}
"""

    doc = "\n//Create new flat array\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "%s bhc_new_A%s(uint64_t size)"%(t['bhc_ary'], t['name'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """
{
    multi_array<%(cpp)s> *ret = new multi_array<%(cpp)s>(size);
    ret->setTemp(false);
    ret->link();
    return (%(bhc_ary)s) ret;
}
"""%t

    doc = "\n//Destroy array\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "void bhc_destroy_A%s(%s ary)"%(t['name'], t['bhc_ary'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """
{
    delete ((multi_array<%(cpp)s>*)ary);
}
"""%t

    doc = "\n//Create view\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "%s bhc_view_A%s("%(t['bhc_ary'], t['name'])
        decl += "const %s src, uint64_t rank, int64_t start, "%t['bhc_ary']
        decl += "const int64_t *shape, const int64_t *stride)"
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """\
{
    bh_base *b = ((multi_array<%(cpp)s>*)src)->meta.base;
    multi_array<%(cpp)s>* ret = new multi_array<%(cpp)s>(b, rank, start, shape, stride);
    ret->setTemp(false);
    return (%(bhc_ary)s) ret;
}
"""%t

    doc = "\n//Get data pointer and:\n"
    doc += "//  if 'force_alloc', force memory allocation before returning the data pointer\n"
    doc += "//  if 'nullify', set the data pointer to NULL after returning the data pointer\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "%s* bhc_data_get_A%s(const %s ary, bh_bool force_alloc, bh_bool nullify)"%(t['bhc'], t['name'], t['bhc_ary'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """\
{
    bh_base *b = ((multi_array<%(cpp)s>*)ary)->meta.base;
    if(force_alloc)
    {
        if(bh_data_malloc(b) != 0)
            return NULL;
    }
    %(bhc)s* ret = (%(bhc)s*)(b->data);
    if(nullify)
        b->data = NULL;
    return ret;
}
"""%t

    doc = "\n//Set data pointer\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "void bhc_data_set_A%(name)s(const %(bhc_ary)s ary, %(bhc)s *data)"%t
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """\
{
    bh_base *b = ((multi_array<%(cpp)s>*)ary)->meta.base;
    b->data = data;
}
"""%t

    doc = "\n//Extension Method, returns 0 when the extension exist\n"
    impl += doc; head += doc
    for key, t in type_map.iteritems():
        decl = "int bhc_extmethod"
        decl += "_A%(name)s_A%(name)s_A%(name)s"%t
        decl += "(const char *name, %(bhc_ary)s out, const %(bhc_ary)s in1, const %(bhc_ary)s in2)"%t
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s\n"%decl
        impl += """
{
    try{
        Runtime::instance().enqueue_extension(name, *((multi_array<%(cpp)s>*) out),
                                                    *((multi_array<%(cpp)s>*) in1),
                                                    *((multi_array<%(cpp)s>*) in2));
    }catch (...){
        return -1;
    }
    return 0;
}
"""%t

    #Let's add header and footer
    head = """/* Bohrium C Bridge: special functions. Auto generated! */

#ifndef __BHC_SPECIALS_H
#define __BHC_SPECIALS_H

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
#include "bhc.h"

using namespace bxx;

%s
"""%impl

    #Finally, let's write the files
    with open(join(args.output,'bhc_specials.h'), 'w') as f:
        f.write(head)
    with open(join(args.output,'bhc_specials.cpp'), 'w') as f:
        f.write(impl)

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

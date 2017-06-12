#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):

    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix,'..','..','core','codegen','types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {'cpp'     : t['cpp'],
                                   'bhc'     : t['bhc'],
                                   'name'    : t['union'],
                                   'bhc_ary' : "bhc_ndarray_%s_p"%t['union']}

    # Let's generate the header and implementation of all data types
    head = ""; impl = ""

    doc = "\n//Flush the Bohrium runtime system\n"
    impl += doc; head += doc
    decl = "void bhc_flush(void)"
    head += "DLLEXPORT %s;\n"%decl
    impl += "%s"%decl
    impl += """
{
    bhxx::Runtime::instance().flush();
}
"""

    doc = "\n//Send and receive a message through the component stack\n"
    doc += "//NB: the returned string is invalidated on the next call to bhc_message()\n"
    impl += doc; head += doc
    decl = "const char* bhc_message(const char* msg)"
    head += "DLLEXPORT %s;\n"%decl
    impl += "%s"%decl
    impl += """
{
    static std::string msg_recv;
    msg_recv = bhxx::Runtime::instance().message(msg);
    return msg_recv.c_str();
}
"""

    doc = "\n//Create new flat array\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "%s bhc_new_A%s(uint64_t size)"%(t['bhc_ary'], t['name'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """
{    
    bhxx::BhArray<%(cpp)s> *ret = new bhxx::BhArray<%(cpp)s>({size});
    return (%(bhc_ary)s) ret;
}
"""%t

    doc = "\n//Destroy array\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_destroy_A%s(%s ary)"%(t['name'], t['bhc_ary'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """
{
    delete ((bhxx::BhArray<%(cpp)s>*)ary);
}
"""%t

    doc = "\n//Create view\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "%s bhc_view_A%s("%(t['bhc_ary'], t['name'])
        decl += "const %s src, uint64_t rank, int64_t start, "%t['bhc_ary']
        decl += "const int64_t *shape, const int64_t *stride)"
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """\
{
    bhxx::Shape _shape(shape, shape+rank);
    bhxx::Stride _stride(stride, stride+rank);
    const auto &b = ((bhxx::BhArray<%(cpp)s>*)src)->base;
    bhxx::BhArray<%(cpp)s>* ret = new bhxx::BhArray<%(cpp)s>(b, _shape, _stride, start);
    return (%(bhc_ary)s) ret;
}
"""%t

    doc = "\n//Get data pointer and:\n"
    doc += "//  if 'force_alloc', force memory allocation before returning the data pointer\n"
    doc += "//  if 'nullify', set the data pointer to NULL after returning the data pointer\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void* bhc_data_get_A%s(const %s ary, bhc_bool force_alloc, bhc_bool nullify)"%(t['name'], t['bhc_ary'])
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """\
{
    bhxx::BhBase* b = ((bhxx::BhArray<%(cpp)s>*)ary)->base.get();

    if(force_alloc) {
        bh_data_malloc(b);
    }

    void* ret = (%(bhc)s*)(b->data);

    if(nullify) {
        b->data = NULL;
    }

    return ret;
}
"""%t

    doc = "\n//Set data pointer\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_data_set_A%(name)s(const %(bhc_ary)s ary, %(bhc)s *data)"%t
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """\
{
    ((bhxx::BhArray<%(cpp)s>*)ary)->base->data = data;
}
"""%t

    doc = "\n//Extension Method, returns 0 when the extension exist\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "int bhc_extmethod"
        decl += "_A%(name)s_A%(name)s_A%(name)s"%t
        decl += "(const char *name, %(bhc_ary)s out, const %(bhc_ary)s in1, const %(bhc_ary)s in2)"%t
        head += "DLLEXPORT %s;\n"%decl
        impl += "%s"%decl
        impl += """
{
    try{
        bhxx::Runtime::instance().enqueue_extmethod(name, *((bhxx::BhArray<%(cpp)s>*) out),
                                                          *((bhxx::BhArray<%(cpp)s>*) in1),
                                                          *((bhxx::BhArray<%(cpp)s>*) in2));
    }catch (... ){
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
#endif // __BHC_SPECIALS_H
"""%head
    impl = """/* Bohrium C Bridge: special functions. Auto generated! */

#include <bhxx/bhxx.hpp>
#include "bhc.h"

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

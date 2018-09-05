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

    doc = "\n// Flush the Bohrium runtime system\n"
    impl += doc; head += doc
    decl = "void bhc_flush(void)"
    head += "%s;\n" % decl
    impl += "%s" % decl
    impl += """
{
    bhxx::Runtime::instance().flush();
}
"""

    doc = "\n// Get the number of times flush has been called\n"
    impl += doc; head += doc
    decl = "int bhc_flush_count(void)"
    head += "%s;\n"%decl
    impl += "%s"%decl
    impl += """
{
    return bhxx::Runtime::instance().getFlushCount();
}
"""

    doc = "\n// Flush and repeat the lazy evaluated operations `nrepeats` times.\n"
    impl += doc; head += doc
    decl = "void bhc_flush_and_repeat(uint64_t nrepeats)"
    head += "%s;\n"%decl
    impl += "%s"%decl
    impl += """
{
    std::shared_ptr<bhxx::BhBase> dummy;
    bhxx::Runtime::instance().flushAndRepeat(nrepeats, dummy);
}
"""

    doc = "\n// Flush and repeat the lazy evaluated operations until `condition` is false or `nrepeats` is reached.\n"
    impl += doc; head += doc
    decl = "void bhc_flush_and_repeat_condition(uint64_t nrepeats, bhc_ndarray_bool8_p condition)"
    head += "%s;\n"%decl
    impl += "%s"%decl
    impl += """
{
    std::shared_ptr<bhxx::BhBase> &b = ((bhxx::BhArray<bool>*)condition)->base;
    bhxx::Runtime::instance().flushAndRepeat(nrepeats, b);
}
"""

    doc = "\n// Send and receive a message through the component stack\n"
    doc += "// NB: the returned string is invalidated on the next call to bhc_message()\n"
    impl += doc; head += doc
    decl = "const char* bhc_message(const char* msg)"
    head += "%s;\n" % decl
    impl += "%s" % decl
    impl += """
{
    static std::string msg_recv;
    msg_recv = bhxx::Runtime::instance().message(msg);
    return msg_recv.c_str();
}
"""

    doc = "\n// Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack.\n"
    doc += "// If the first VE isn't a device, NULL is returned.\n"
    impl += doc; head += doc
    decl = "void* bhc_getDeviceContext(void)"
    head += "%s;\n" % decl
    impl += "%s" % decl
    impl += """
{
    return bhxx::Runtime::instance().getDeviceContext();
}
"""

    doc = "\n// Set the context handle, such as CUDA's context, of the first VE in the runtime stack.\n"
    doc += "// If the first VE isn't a device, nothing happens.\n"
    impl += doc; head += doc
    decl = "void bhc_set_device_context(uint64_t device_context)"
    head += "%s;\n" % decl
    impl += "%s" % decl
    impl += """
{
    bhxx::Runtime::instance().setDeviceContext((void*)device_context);
}
"""

    doc = "\n// Create new flat array\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "%s bhc_new_A%s(uint64_t size)"%(t['bhc_ary'], t['name'])
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """
{
    bhxx::BhArray<%(cpp)s> *ret = new bhxx::BhArray<%(cpp)s>({size});
    return (%(bhc_ary)s) ret;
}

""" % t

    doc = "\n// Destroy array\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_destroy_A%s(%s ary)"%(t['name'], t['bhc_ary'])
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """
{
    delete ((bhxx::BhArray<%(cpp)s>*)ary);
}

""" % t

    doc = "\n// Create view\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "%s bhc_view_A%s(" % (t['bhc_ary'], t['name'])
        decl += "const %s src, uint64_t rank, int64_t start, " % t['bhc_ary']
        decl += "const int64_t *shape, const int64_t *stride)"
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
    bhxx::Shape _shape(shape, shape+rank);
    bhxx::Stride _stride(stride, stride+rank);
    bhxx::BhArray<%(cpp)s>* ret = new bhxx::BhArray<%(cpp)s>(
        ((bhxx::BhArray<%(cpp)s>*)src)->base,
        _shape,
        _stride,
        start
    );
    return (%(bhc_ary)s) ret;
}

""" % t

    doc = "\n// Get data pointer from the first VE in the runtime stack\n"
    doc += "//   if 'copy2host', always copy the memory to main memory\n"
    doc += "//   if 'force_alloc', force memory allocation before returning the data pointer\n"
    doc += "//   if 'nullify', set the data pointer to NULL after returning the data pointer\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void* bhc_data_get_A%s(const %s ary, bhc_bool copy2host, bhc_bool force_alloc, bhc_bool nullify)" % (t['name'], t['bhc_ary'])
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
    std::shared_ptr<bhxx::BhBase> &b = ((bhxx::BhArray<%(cpp)s>*)ary)->base;
    return bhxx::Runtime::instance().getMemoryPointer(b, copy2host, force_alloc, nullify);
}

""" % t

    doc = "\n// Set data pointer in the first VE in the runtime stack\n"
    doc += "// NB: The component will deallocate the memory when encountering a BH_FREE\n"
    doc += "//   if 'host_ptr', the pointer points to the host memory (main memory) as opposed to device memory\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_data_set_A%(name)s(const %(bhc_ary)s ary, bhc_bool host_ptr, %(bhc)s *data)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
   std::shared_ptr<bhxx::BhBase> &b = ((bhxx::BhArray<%(cpp)s>*)ary)->base;
   bhxx::Runtime::instance().setMemoryPointer(b, host_ptr, data);
}

""" % t

    doc = "\n// Copy the memory of `src` to `dst`\n"
    doc += "//   Use 'param' to set compression parameters or use the empty string\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_data_copy_A%(name)s(const %(bhc_ary)s src, const %(bhc_ary)s dst, const char *param)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
    bhxx::Runtime::instance().memCopy(*((bhxx::BhArray<%(cpp)s>*) src), *((bhxx::BhArray<%(cpp)s>*) dst),
                                      std::string{param});
}

""" % t

    doc = "\n// Informs the runtime system to make data synchronized and available after the next flush().\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_sync_A%(name)s(const %(bhc_ary)s ary)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
    std::shared_ptr<bhxx::BhBase> &b = ((bhxx::BhArray<%(cpp)s>*)ary)->base;
    bhxx::Runtime::instance().sync(b);
}

""" % t

    doc = "\n// Slides the view of an array in the given dimensions, by the given strides for each iteration in a loop.\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_slide_view"
        decl += "_A%(name)s" % t
        decl += "(const %(bhc_ary)s ary1, int64_t dim, int64_t slide, int64_t view_shape, \
                 int64_t array_shape, int64_t array_stride, int64_t step_delay)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
    bhxx::Runtime::instance().slide_view(
        (bhxx::BhArray<%(cpp)s>*) ary1,
        dim,
        slide,
        view_shape,
        array_shape,
        array_stride,
        step_delay);
}

""" % t

    doc = "\n// Set a reset for an iterator in a dynamic view within a loop.\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "void bhc_add_reset"
        decl += "_A%(name)s" % t
        decl += "(const %(bhc_ary)s ary1, int64_t dim, int64_t reset_max)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """\
{
   bhxx::Runtime::instance().add_reset(
        (bhxx::BhArray<%(cpp)s>*) ary1,
        dim,
        reset_max);
}

""" % t

    doc = "\n// Extension Method, returns 0 when the extension exist\n"
    impl += doc; head += doc
    for key, t in type_map.items():
        decl = "int bhc_extmethod"
        decl += "_A%(name)s_A%(name)s_A%(name)s" % t
        decl += "(const char *name, %(bhc_ary)s out, const %(bhc_ary)s in1, const %(bhc_ary)s in2)" % t
        head += "%s;\n" % decl
        impl += "%s" % decl
        impl += """
{
    try {
        bhxx::Runtime::instance().enqueueExtmethod(
            name,
            *((bhxx::BhArray<%(cpp)s>*) out),
            *((bhxx::BhArray<%(cpp)s>*) in1),
            *((bhxx::BhArray<%(cpp)s>*) in2)
        );
    } catch (...) {
        return -1;
    }
    return 0;
}

""" % t

    #Let's add header and footer
    head = """/* Bohrium C Bridge: special functions. Auto generated! */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

%s
#ifdef __cplusplus
}
#endif

""" % head
    impl = """/* Bohrium C Bridge: special functions. Auto generated! */

#include <bhxx/bhxx.hpp>
#include "bhc.h"

%s
""" % impl

    #Finally, let's write the files
    with open(join(args.output, 'bhc_specials.h'), 'w') as f:
        f.write(head)
    with open(join(args.output, 'bhc_specials.cpp'), 'w') as f:
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

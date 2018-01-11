/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <Python.h>
#include <structmember.h>
#include <bhc.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
    #define NPY_PY3K
#endif

typedef struct {
    PyObject_HEAD
    Py_ssize_t size;
    PyArray_Descr *dtype;
    PyObject *dtype_name;
    PyObject *bhc_ary_swig_ptr;
    void *bhc_ary_ptr;
} BhBase;

static PyMemberDef BhBase_members[] = {
    {"size", T_PYSSIZET, offsetof(BhBase, size), READONLY, "Number of elements"},
    {"dtype", T_OBJECT, offsetof(BhBase, dtype), READONLY, "The dtype"},
    {"dtype_name", T_OBJECT, offsetof(BhBase, dtype_name), READONLY, "The dtype name"},
    {"bhc_obj", T_OBJECT, offsetof(BhBase, bhc_ary_swig_ptr), 0, "A SWIG pointer to the array"},
    {NULL}  /* Sentinel */
};

static bhc_dtype dtype_np2bhc(const PyArray_Descr *dtype) {
    switch(dtype->type_num) {
        case NPY_BOOL:
            return BH_BOOL;
        case NPY_INT8:
            return BH_INT8;
        case NPY_INT16:
            return BH_INT16;
        case NPY_INT32:
            return BH_INT32;
        case NPY_INT64:
            return BH_INT64;
        case NPY_UINT8:
            return BH_UINT8;
        case NPY_UINT16:
            return BH_UINT16;
        case NPY_UINT32:
            return BH_UINT32;
        case NPY_UINT64:
            return BH_UINT64;
        case NPY_FLOAT32:
            return BH_FLOAT32;
        case NPY_FLOAT64:
            return BH_FLOAT64;
        case NPY_COMPLEX64:
            return BH_COMPLEX64;
        case NPY_COMPLEX128:
            return BH_COMPLEX128;
        default:
            fprintf(stderr, "dtype_np2bhc() - unknown dtype!\n");
            assert(1==2);
            exit(-1);
    }
}

static PyObject* BhBase_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BhBase *self;
    self = (BhBase *)type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static int
BhBase_init(BhBase *self, PyObject *args, PyObject *kwds) {
    Py_ssize_t size = 0;
    PyArray_Descr *dtype = NULL;
    PyObject *bhc_ary_swig_ptr = NULL;
    PyObject *dtype_name = NULL;
    static char *kwlist[] = {"size", "dtype", "dtype_name", "bhc_ary_swig_ptr", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "nO&OO", kwlist,
                                    &size,
                                    PyArray_DescrConverter, &dtype,
                                    &dtype_name,
                                    &bhc_ary_swig_ptr)) {
        return -1;
    }
    // Empty bases cannot have any `bhc_ary_swig_ptr`
    assert(size > 0 || bhc_ary_swig_ptr == Py_None);
    self->size = size;
    self->dtype = dtype;
    self->dtype_name = dtype_name;
    Py_INCREF(self->dtype_name);
    self->bhc_ary_swig_ptr = bhc_ary_swig_ptr;
    Py_INCREF(self->bhc_ary_swig_ptr);
    if (size > 0) {
        PyObject *t = PyObject_CallMethod(bhc_ary_swig_ptr, "__int__", NULL);
        self->bhc_ary_ptr = PyLong_AsVoidPtr(t);
        Py_XDECREF(t);
    }
    return 0;
}

static void BhBase_dealloc(BhBase* self) {

    if (self->size > 0) {
        bhc_destroy(dtype_np2bhc(self->dtype), self->bhc_ary_ptr);
    }

    Py_XDECREF(self->dtype);
    Py_XDECREF(self->dtype_name);
    Py_XDECREF(self->bhc_ary_swig_ptr);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyTypeObject BhBaseType = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                              // ob_size
#endif
    "_bhc_base_array.BhBase",         // tp_name
    sizeof(BhBase),                 // tp_basicsize
    0,                              // tp_itemsize
    (destructor)BhBase_dealloc,     // tp_dealloc
    0,                              // tp_print
    0,                              // tp_getattr
    0,                              // tp_setattr
#if defined(NPY_PY3K)
    0,                              // tp_reserved
#else
    0,                              // tp_compare
#endif
    0,                              // tp_repr
    0,                              // tp_as_number
    0,                              // tp_as_sequence
    0,                              // tp_as_mapping
    0,                              // tp_hash
    0,                              // tp_call
    0,                              // tp_str
    0,                              // tp_getattro
    0,                              // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,            // tp_flags
    0,                              // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear
    0, // tp_richcompare
    0,                              // tp_weaklistoffset
    0,                              // tp_iter
    0,                              // tp_iternext
    0,                               // tp_methods
    BhBase_members,                 // tp_members
    0,                              // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    0,                              // tp_dictoffset
    (initproc) BhBase_init,         // tp_init
    0,                  // tp_alloc
    BhBase_new,                     // tp_new
    0,         // tp_free
    0,                              // tp_is_gc
    0,                              // tp_bases
    0,                              // tp_mro
    0,                              // tp_cache
    0,                              // tp_subclasses
    0,                              // tp_weaklist
    0,                              // tp_del
    0,                              // tp_version_tag
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_bhc_base_array",
        NULL,
        -1,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__bhc_base_array(void)
#else
#define RETVAL
PyMODINIT_FUNC init_bhc_base_array(void)
#endif
{
    PyObject *m;
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_bhc_base_array", NULL);
#endif
    if (m == NULL) {
        return RETVAL;
    }

    // Import NumPy
    import_array();
    if (PyType_Ready(&BhBaseType) < 0) {
        return RETVAL;
    }
    PyModule_AddObject(m, "BhBase", (PyObject*) &BhBaseType);
    return RETVAL;
}

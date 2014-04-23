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
#include <dlfcn.h>
#include <bh.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

//The NumPy API changed in version 1.7
#if(NPY_API_VERSION >= 0x00000007)
    #define BH_PyArrayObject PyArrayObject_fields
#else
    #define BH_PyArrayObject PyArrayObject
    #define NPY_ARRAY_OWNDATA NPY_OWNDATA
#endif

//Forward declaration
static PyObject *BhArray_data_bhc2np(PyObject *self, PyObject *args);
static PyTypeObject BhArrayType;

#define BhArray_CheckExact(op) (((PyObject*)(op))->ob_type == &BhArrayType)
PyObject *ndarray = NULL; //The ndarray Python module
PyObject *ufunc = NULL; //The ufunc Python module
PyObject *bohrium = NULL; //The Bohrium Python module
PyObject *array_create = NULL; //The array_create Python module

typedef struct
{
    BH_PyArrayObject base;
    PyObject *bhc_ary;
    PyObject *array_priority;
}BhArray;

//Help function to retrieve the Bohrium-C data pointer
//Return -1 on error
static int get_bhc_data_pointer(PyObject *ary, int force_allocation, int nullify, void **out_data)
{
    PyObject *data = PyObject_CallMethod(ndarray, "get_bhc_data_pointer",
                                         "Oii", ary, force_allocation, nullify);
    if(data == NULL)
        return -1;
    if(!PyInt_Check(data))
    {
        PyErr_SetString(PyExc_TypeError, "get_bhc_data_pointer(ary) should "
                "return a Python integer that represents a memory address");
        Py_DECREF(data);
        return -1;
    }
    void *d = PyLong_AsVoidPtr(data);
    Py_DECREF(data);
    if(force_allocation && d == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "get_bhc_data_pointer(ary, allocate=True) "
                                         "shouldn't return a NULL pointer");
        return -1;
    }
    *out_data = d;
    return 0;
}

//Help function for memory re-map
//Return -1 on error
static int _mremap_data(void *dst, void *src, bh_intp size)
{
#if MREMAP_FIXED
    if(mremap(src, size, size, MREMAP_FIXED|MREMAP_MAYMOVE, dst) == MAP_FAILED)
    {
        int errsv = errno;//mremap() sets the errno.
        PyErr_Format(PyExc_RuntimeError,"Error - could not mremap a "
                     "data region. Returned error code by mremap(): %s.\n"
                     ,strerror(errsv));
        return -1;
    }
    return 0;
#else
    //Systems that doesn't support mremap will use memcpy, which introduces a
    //race-condition if another thread access the 'dst' memory before memcpy finishes.
    if(mprotect(dst, size, PROT_WRITE) != 0)
    {
        int errsv = errno;//mprotect() sets the errno.
        PyErr_Format(PyExc_RuntimeError,"Error - could not (un-)mprotect a "
                     "data region. Returned error code by mprotect(): %s.\n"
                     ,strerror(errsv));
        return -1;
    }
    memcpy(dst, src, size);
    return munmap(src, size);
#endif
}

//Callback function called by the signal library
void mem_access_callback(unsigned long id, uintptr_t addr)
{
    PyObject *ary = (PyObject *) id;
    printf("mem_access_callback() - ary: %p, addr: %p\n", ary, (void*) addr);

    PyGILState_STATE GIL = PyGILState_Ensure();
    PyErr_WarnEx(NULL,"Encountering an operation not supported by Bohrium. "
                      "It will be handled by the original NumPy.",1);

    if(BhArray_data_bhc2np(ary, NULL) == NULL)
        PyErr_Print();
    PyGILState_Release(GIL);
}

//Help function for protecting the memory of the NumPy part of 'ary'
//Return -1 on error
static int _mprotect_np_part(BhArray *ary)
{
    //Finally we memory protect the NumPy data
    if(mprotect(ary->base.data, PyArray_NBYTES((PyArrayObject*)ary), PROT_NONE) == -1)
    {
        int errsv = errno;//mprotect() sets the errno.
        PyErr_Format(PyExc_RuntimeError,"Error - could not protect a data"
                     "data region. Returned error code by mprotect: %s.\n",
                     strerror(errsv));
        return -1;
    }
    attach_signal((signed long)ary, (uintptr_t) ary->base.data,
                  PyArray_NBYTES((PyArrayObject*)ary), mem_access_callback);
    return 0;
}

//Help function for allocate protected memory for the NumPy part of 'ary'
//Return -1 on error
static int _protected_malloc(BhArray *ary)
{
    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *addr = mmap(0, PyArray_NBYTES((PyArrayObject*)ary), PROT_NONE,
                      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(addr == MAP_FAILED)
    {
        int errsv = errno;//mmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mmap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        return -1;
    }
    //Update the ary data pointer.
    ary->base.data = addr;

    attach_signal((signed long)ary, (uintptr_t) ary->base.data,
                  PyArray_NBYTES((PyArrayObject*)ary), mem_access_callback);
    return 0;
}

static PyObject *
BhArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BhArray *ret = (BhArray *) PyArray_Type.tp_new(type, args, kwds);
    if(!PyArray_CHKFLAGS((PyArrayObject*)ret, NPY_ARRAY_OWNDATA))
        return (PyObject *) ret;//The array doesn't own the array data

    //Lets free the NumPy allocated memory and allocate/mprotect instead
    free(ret->base.data);

    if(_protected_malloc(ret) != 0)
        return NULL;

    return (PyObject *) ret;
}

static PyObject *
BhArray_alloc(PyTypeObject *type, Py_ssize_t nitems)
{
    PyObject *obj;
    obj = (PyObject *)malloc(type->tp_basicsize);
    PyObject_Init(obj, type);
    return obj;
}

static void
BhArray_dealloc(BhArray* self)
{
    assert(BhArray_CheckExact(self));

    if(self->bhc_ary == NULL)
        goto finish;

    if(self->bhc_ary == Py_None)
    {
        Py_DECREF(Py_None);
        goto finish;
    }
    PyObject *r = PyObject_CallMethod(ndarray, "del_bhc_obj", "O", self->bhc_ary);
    if(r == NULL)
    {
        PyErr_Print();
        goto finish;
    }
finish:
    if(!PyArray_CHKFLAGS((PyArrayObject*)self, NPY_ARRAY_OWNDATA))
    {
        BhArrayType.tp_base->tp_dealloc((PyObject*)self);
        return;//The array doesn't own the array data
    }

    assert(!PyDataType_FLAGCHK(PyArray_DESCR((PyArrayObject*)self), NPY_ITEM_REFCOUNT));

    void *addr = PyArray_DATA((PyArrayObject*)self);
    if(munmap(addr, PyArray_NBYTES((PyArrayObject*)self)) == -1)
    {
        int errsv = errno;//munmmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mummap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        PyErr_Print();
    }
    detach_signal((signed long)self, mem_access_callback);
    self->base.data = NULL;
    BhArrayType.tp_base->tp_dealloc((PyObject*)self);
}

static PyObject *
BhArray_finalize(PyObject *self, PyObject *args)
{
    int e = PyObject_IsInstance(self, (PyObject*) &BhArrayType);
    if(e == -1)
    {
        return NULL;
    }
    else if (e == 0)
    {
        Py_RETURN_NONE;
    }
    ((BhArray*)self)->bhc_ary = Py_None;
    //The __array_priority__ should be greater than 0.0 to give Bohrium precedence
    ((BhArray*)self)->array_priority = PyFloat_FromDouble(2.0);
    Py_INCREF(Py_None);
    Py_RETURN_NONE;
}

static PyObject *
BhArray_data_bhc2np(PyObject *self, PyObject *args)
{
    assert(args == NULL);
    assert(BhArray_CheckExact(self));

    //We move the whole array (i.e. the base array) from Bohrium to NumPy
    PyObject *base = PyObject_CallMethod(ndarray, "get_base", "O", self);
    if(base == NULL)
        return NULL;
    assert(BhArray_CheckExact(base));

/* TODO: handle the case where bhc_ary is None by unprotecting the memory.
    //Check if we need to do anything
    if(((BhArray*)base)->bhc_ary == Py_None)
    {
        Py_DECREF(base);
        Py_RETURN_NONE;
    }
*/
    //Calling get_bhc_data_pointer(base, allocate=False)
    void *d = NULL;
    if(get_bhc_data_pointer(base, 0, 1, &d) == -1)
        return NULL;
    Py_DECREF(base);
    if(d != NULL)
    {
        if(_mremap_data(PyArray_DATA((PyArrayObject*)base), d,
                        PyArray_NBYTES((PyArrayObject*)base)) != 0)
            return NULL;
    }
    detach_signal((signed long)base, mem_access_callback);

    //Lets delete the current bhc_ary
    if(PyObject_CallMethod(ndarray, "del_bhc", "O", self) == NULL)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *
BhArray_data_np2bhc(PyObject *self, PyObject *args)
{
    assert(args == NULL);
    assert(BhArray_CheckExact(self));

    //We move the whole array (i.e. the base array) from Bohrium to NumPy
    PyObject *base = PyObject_CallMethod(ndarray, "get_base", "O", self);
    if(base == NULL)
        return NULL;
    assert(BhArray_CheckExact(base));

    //Make sure that bhc_ary exist
    if(((BhArray*)base)->bhc_ary == Py_None)
    {
        PyObject *err = PyObject_CallMethod(ndarray, "new_bhc_base", "O", base);
        if(err == NULL)
            return NULL;
        Py_DECREF(err);
    }

    //Calling get_bhc_data_pointer(base, allocate=True)
    void *d = NULL;
    if(get_bhc_data_pointer(base, 1, 0, &d) == -1)
        return NULL;
    Py_DECREF(base);
    if(d != NULL)
    {
        memcpy(d, PyArray_DATA((PyArrayObject*)base), PyArray_NBYTES((PyArrayObject*)base));
        if(_mprotect_np_part((BhArray*)base) != 0)
            return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
BhArray_data_fill(PyObject *self, PyObject *args)
{
    PyObject *np_ary;
    if(!PyArg_ParseTuple(args, "O:ndarray", &np_ary))
        return NULL;

    if(!PyArray_Check(np_ary))
    {
        PyErr_SetString(PyExc_TypeError, "must be a NumPy array");
        return NULL;
    }

    if(!PyArray_ISCARRAY_RO((PyArrayObject*)np_ary))
    {
        PyErr_SetString(PyExc_TypeError, "must be a C-style contiguous array");
        return NULL;
    }

    //Calling get_bhc_data_pointer(self, allocate=True)
    void *d = NULL;
    if(get_bhc_data_pointer(self, 1, 0, &d) == -1)
        return NULL;

    memcpy(d, PyArray_DATA((PyArrayObject*)np_ary), PyArray_NBYTES((PyArrayObject*)np_ary));

    Py_RETURN_NONE;
}

static PyObject *
BhArray_copy(PyObject *self, PyObject *args)
{
    assert(args == NULL);
    PyObject *ret = PyObject_CallMethod(array_create, "empty_like", "O", self);
    if(ret == NULL)
        return NULL;
    PyObject *err = PyObject_CallMethod(ufunc, "assign", "OO", self, ret);
    if(err == NULL)
    {
        Py_DECREF(ret);
        return NULL;
    }
    Py_DECREF(err);
    return ret;
}

static PyObject *
BhArray_copy2numpy(PyObject *self, PyObject *args)
{
    assert(args == NULL);

    PyObject *ret = PyArray_NewLikeArray((PyArrayObject*)self, NPY_ANYORDER, NULL, 0);
    if(ret == NULL)
        return NULL;
    PyObject *err = PyObject_CallMethod(ufunc, "assign", "OO", self, ret);
    if(err == NULL)
    {
        Py_DECREF(ret);
        return NULL;
    }
    Py_DECREF(err);
    return ret;
}

static PyMethodDef BhArrayMethods[] = {
    {"__array_finalize__", BhArray_finalize, METH_VARARGS, NULL},
    {"_data_bhc2np", BhArray_data_bhc2np, METH_NOARGS, "Copy the Bohrium-C data to NumPy data"},
    {"_data_np2bhc", BhArray_data_np2bhc, METH_NOARGS, "Copy the NumPy data to Bohrium-C data"},
    {"_data_fill", BhArray_data_fill, METH_VARARGS, "Fill the Bohrium-C data from a numpy NumPy"},
    {"copy", BhArray_copy, METH_NOARGS, "Copy the array in C-style memory layout"},
    {"copy2numpy", BhArray_copy2numpy, METH_NOARGS, "Copy the array in C-style memory "
                                                    "layout to a regular NumPy array"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyObject *
BhArray_get_bhc_ary(BhArray *self, void *closure)
{
    Py_INCREF(self->bhc_ary);
    return self->bhc_ary;
}
static int
BhArray_set_bhc_ary(BhArray *self, PyObject *value, void *closure)
{
    Py_INCREF(value);
    self->bhc_ary = value;
    return 0;

}
static PyObject *
BhArray_get_ary_pri(BhArray *self, void *closure)
{
    Py_INCREF(self->array_priority);
    return self->array_priority;
}
static int
BhArray_set_ary_pri(BhArray *self, PyObject *value, void *closure)
{
    Py_INCREF(value);
    self->array_priority = value;
    return 0;
}
static PyGetSetDef BhArray_getseters[] = {
    {"bhc_ary",
     (getter)BhArray_get_bhc_ary,
     (setter)BhArray_set_bhc_ary,
     "The Bohrium C-Bridge array",
     NULL},
    {"__array_priority__",
     (getter)BhArray_get_ary_pri,
     (setter)BhArray_set_ary_pri,
     "The NumPy / Bohrium array precedence",
     NULL},
    {NULL}  /* Sentinel */
};

static int
BhArray_SetItem(PyObject *o, PyObject *key, PyObject *v)
{
    if(v == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }
    if(!PyArray_ISWRITEABLE((PyArrayObject *)o))
    {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }

    PyObject *view = PyArray_Type.tp_as_mapping->mp_subscript(o, key);
    if(view == NULL)
        return -1;

    PyObject *ret = PyObject_CallMethod(ufunc, "assign", "OO", v, view);
    Py_DECREF(view);
    if(ret == NULL)
        return -1;

    return 0;
}

static int
BhArray_SetSlice(PyObject *o, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v)
{
    if(v == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }
    if(!PyArray_ISWRITEABLE((PyArrayObject *)o))
    {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }

    PyObject *view = PyArray_Type.tp_as_sequence->sq_slice(o, ilow, ihigh);
    if(view == NULL)
        return -1;

    PyObject *ret = PyObject_CallMethod(ufunc, "assign", "OO", v, view);
    Py_DECREF(view);
    if(ret == NULL)
        return -1;
    return 0;
}

static PyObject *
BhArray_GetItem(PyObject *o, PyObject *k)
{
    Py_ssize_t i;
    //If the result is a scalar we let NumPy handle it

    //If the tuple access all dimensions we must check for Python slice objects
    if(PyTuple_Check(k) && (PyTuple_GET_SIZE(k) == PyArray_NDIM((PyArrayObject*)o)))
    {
        for(i=0; i<PyTuple_GET_SIZE(k); ++i)
        {
            PyObject *a = PyTuple_GET_ITEM(k, i);
            if(PySlice_Check(a))
            {
                //A slice never results in a scalar output
                return PyArray_Type.tp_as_mapping->mp_subscript(o, k);
            }
        }
    }
    if(PyArray_IsIntegerScalar(k) || (PyIndex_Check(k) && !PySequence_Check(k)))
    {
        if(BhArray_data_bhc2np(o, NULL) == NULL)
            return NULL;
    }
    return PyArray_Type.tp_as_mapping->mp_subscript(o, k);
}

static PyMappingMethods array_as_mapping = {
    (lenfunc)0,                     /*mp_length*/
    (binaryfunc)BhArray_GetItem,    /*mp_subscript*/
    (objobjargproc)BhArray_SetItem, /*mp_ass_subscript*/
};
static PySequenceMethods array_as_sequence = {
    (lenfunc)0,                              /*sq_length*/
    (binaryfunc)NULL,                        /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,                      /*sq_repeat*/
    (ssizeargfunc)BhArray_GetItem,           /*sq_item*/
    (ssizessizeargfunc)0,                    /*sq_slice (Not in the Python doc)*/
    (ssizeobjargproc)BhArray_SetItem,        /*sq_ass_item*/
    (ssizessizeobjargproc)BhArray_SetSlice,  /*sq_ass_slice (Not in the Python doc)*/
    (objobjproc) 0,                          /*sq_contains */
    (binaryfunc) NULL,                       /*sg_inplace_concat */
    (ssizeargfunc)NULL,                      /*sg_inplace_repeat */
};

static PyObject *
BhArray_Repr(PyObject *self)
{
    assert(BhArray_CheckExact(self));
    PyObject *t = BhArray_copy2numpy(self, NULL);
    if(t == NULL)
        return NULL;
    PyObject *str = PyArray_Type.tp_repr(self);
    Py_DECREF(t);
    return str;
}
static PyObject *
BhArray_Str(PyObject *self)
{
    assert(BhArray_CheckExact(self));
    PyObject *t = BhArray_copy2numpy(self, NULL);
    if(t == NULL)
        return NULL;
    PyObject *str = PyArray_Type.tp_str(self);
    Py_DECREF(t);
    return str;
}

//Importing the array_as_number struct
#include "operator_overload.c"

static PyTypeObject BhArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                       /* ob_size */
    "bohrium.ndarray",       /* tp_name */
    sizeof(BhArray),         /* tp_basicsize */
    0,                       /* tp_itemsize */
    (destructor) BhArray_dealloc,/* tp_dealloc */
    0,                       /* tp_print */
    0,                       /* tp_getattr */
    0,                       /* tp_setattr */
    0,                       /* tp_compare */
    &BhArray_Repr,           /* tp_repr */
    &array_as_number,        /* tp_as_number */
    &array_as_sequence,      /* tp_as_sequence */
    &array_as_mapping,       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    &BhArray_Str,            /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
      Py_TPFLAGS_BASETYPE |
      Py_TPFLAGS_CHECKTYPES, /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    (richcmpfunc)array_richcompare, /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    BhArrayMethods,          /* tp_methods */
    0,                       /* tp_members */
    BhArray_getseters,       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    0,                       /* tp_init */
    BhArray_alloc,           /* tp_alloc */
    BhArray_new,             /* tp_new */
};

PyMODINIT_FUNC
init_bh(void)
{
    PyObject *m;

    m = Py_InitModule("_bh", NULL);
    if (m == NULL)
        return;

    //Import NumPy
    import_array();

    BhArrayType.tp_base = &PyArray_Type;
    if (PyType_Ready(&BhArrayType) < 0)
        return;

    PyModule_AddObject(m, "ndarray", (PyObject *)&BhArrayType);

    ndarray = PyImport_ImportModule("bohrium.ndarray");
    if(ndarray == NULL)
        return;
    ufunc = PyImport_ImportModule("bohrium.ufunc");
    if(ufunc == NULL)
        return;
    bohrium = PyImport_ImportModule("bohrium");
    if(bohrium == NULL)
        return;
    array_create = PyImport_ImportModule("bohrium.array_create");
    if(array_create == NULL)
        return;

    //Initialize the signal handler
    init_signal();
}

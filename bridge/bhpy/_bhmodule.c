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

#define BhArray_CheckExact(op) (((PyObject*)(op))->ob_type == &BhArrayType)
static PyTypeObject BhArrayType;
PyObject *ndarray = NULL; //The ndarray Python module
PyObject *ufunc = NULL; //The ufunc Python module
PyObject *bohrium = NULL; //The Bohrium Python module

typedef struct
{
    PyArrayObject_fields base;
    PyObject *bhc_ary;
}BhArray;

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
    PyObject_Del(self);
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
    Py_INCREF(Py_None);
    Py_RETURN_NONE;
}

static PyObject *
BhArray_data_bhc2np(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ""))
        return NULL;

    //We move the whole array (i.e. the base array) from Bohrium to NumPy
    PyObject *base = PyObject_CallMethod(ndarray, "get_base", "O", self);
    if(base == NULL)
        return NULL;

    PyObject *data = PyObject_CallMethod(ndarray, "get_bhc_data_pointer", "O", base);
    if(data == NULL)
        return NULL;

    if(!PyInt_Check(data))
    {
        PyErr_SetString(PyExc_TypeError, "get_bhc_data_pointer(ary) should "
                "return a Python integer that represents a memory address");
        return NULL;
    }

    //Lets copy data
    void *d = PyLong_AsVoidPtr(data);
    if(d != NULL)
    {
        memcpy(PyArray_DATA((PyArrayObject*)base), d, PyArray_NBYTES((PyArrayObject*)base));
    }

    //Lets delete the current bhc_ary
    return PyObject_CallMethod(ndarray, "del_bhc", "O", base);
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
    PyObject *data = PyObject_CallMethod(ndarray, "get_bhc_data_pointer", "Oi", self,1);
    if(data == NULL)
        return NULL;
    if(!PyInt_Check(data))
    {
        PyErr_SetString(PyExc_TypeError, "get_bhc_data_pointer(ary) should "
                "return a Python integer that represents a memory address");
        return NULL;
    }
    void *d = PyLong_AsVoidPtr(data);
    if(d == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "get_bhc_data_pointer(ary, allocate=True) "
                                         "shouldn't return a NULL pointer");
        return NULL;
    }

    memcpy(d, PyArray_DATA((PyArrayObject*)np_ary), PyArray_NBYTES((PyArrayObject*)np_ary));

    Py_RETURN_NONE;
}

static PyMethodDef BhArrayMethods[] = {
    {"__array_finalize__", BhArray_finalize, METH_VARARGS, NULL},
    {"_data_bhc2np", BhArray_data_bhc2np, METH_VARARGS, "Copy the Bohrium-C data to NumPy data"},
    {"_data_fill", BhArray_data_fill, METH_VARARGS, "Fill the Bohrium-C data from a numpy NumPy"},
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
static PyGetSetDef BhArray_getseters[] = {
    {"bhc_ary",
     (getter)BhArray_get_bhc_ary,
     (setter)BhArray_set_bhc_ary,
     "The Bohrium C-Bridge array",
     NULL},
    {NULL}  /* Sentinel */
};

static int
BhArray_SetItem(PyObject *o, PyObject *key, PyObject *v)
{
    PyObject *view = PyArray_Type.tp_as_mapping->mp_subscript(o, key);
    if(view == NULL)
        return -1;

    PyObject *ret = PyObject_CallMethod(ndarray, "assign", "OO", v, view);
    if(ret == NULL)
    {
        Py_DECREF(view);
        return -1;
    }
    Py_DECREF(view);
    Py_DECREF(ret);
    return 0;
}

static int
BhArray_SetSlice(PyObject *o, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v)
{
    PyObject *view = PyArray_Type.tp_as_sequence->sq_slice(o, ilow, ihigh);
    if(view == NULL)
        return -1;

    PyObject *ret = PyObject_CallMethod(ufunc, "assign", "OO", v, view);
    if(ret == NULL)
    {
        Py_DECREF(view);
        return -1;
    }
    Py_DECREF(view);
    Py_DECREF(ret);
    return 0;
}

static PyMappingMethods array_as_mapping = {
    (lenfunc)0,                     /*mp_length*/
    (binaryfunc)0,                  /*mp_subscript*/
    (objobjargproc)BhArray_SetItem, /*mp_ass_subscript*/
};
static PySequenceMethods array_as_sequence = {
    (lenfunc)0,                              /*sq_length*/
    (binaryfunc)NULL,                        /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,                      /*sq_repeat*/
    (ssizeargfunc)0,                         /*sq_item*/
    (ssizessizeargfunc)0,                    /*sq_slice (Not in the Python doc)*/
    (ssizeobjargproc)BhArray_SetItem,        /*sq_ass_item*/
    (ssizessizeobjargproc)BhArray_SetSlice,  /*sq_ass_slice (Not in the Python doc*/
    (objobjproc) 0,                          /*sq_contains */
    (binaryfunc) NULL,                       /*sg_inplace_concat */
    (ssizeargfunc)NULL,                      /*sg_inplace_repeat */
};

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
    0,                       /* tp_repr */
    &array_as_number,        /* tp_as_number */
    &array_as_sequence,      /* tp_as_sequence */
    &array_as_mapping,       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    0,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
      Py_TPFLAGS_BASETYPE |
      Py_TPFLAGS_CHECKTYPES, /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
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
    0,                       /* tp_alloc */
    0,                       /* tp_new */
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
    if(ndarray == NULL)
        return;
    bohrium = PyImport_ImportModule("bohrium");
    if(bohrium == NULL)
        return;
}

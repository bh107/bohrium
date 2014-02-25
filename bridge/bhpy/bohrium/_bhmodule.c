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
        return;

    if(self->bhc_ary == Py_None)
    {
        Py_DECREF(Py_None);
        return;
    }

    PyObject *m = PyImport_ImportModule("_util");
    if(m == NULL)
    {
        PyErr_Print();
        return;
    }
    PyObject *arys_to_destory = PyObject_GetAttrString(m,"bhc_arys_to_destroy");
    if(arys_to_destory == NULL)
    {
        PyErr_Print();
        return;
    }
    if(PyList_Append(arys_to_destory, self->bhc_ary) != 0)
    {
        PyErr_Print();
        return;
    }
    Py_DECREF(m);
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

static PyMethodDef BhArrayMethods[] = {
    {"__array_finalize__", BhArray_finalize, METH_VARARGS, NULL},
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
    0,                       /* tp_as_number */
    0,                       /* tp_as_sequence */
    0,                       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    0,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
      Py_TPFLAGS_BASETYPE,   /* tp_flags */
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
}

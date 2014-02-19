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

#include "types.h"
#include "types.c"

#define BhArray_CheckExact(op) (((PyObject*)(op))->ob_type == &BhArrayType)

static void *bhc_handle;
static PyObject *dtype_npy2bh=NULL;

static PyObject *
dtype_set_map(PyObject *self, PyObject *args)
{
    PyObject *map;

    if(!PyArg_ParseTuple(args, "O", &map))
        return NULL;

    if(!PyDict_Check(map))
    {
        PyErr_Format(PyExc_TypeError, "The argument must be a dict that maps "
            "data types from NumPy to Bohrium (e.g. NPY_FLOAT32 to BH_FLOAT32)");
        return NULL;
    }
    Py_XDECREF(map);
    dtype_npy2bh = map;

    Py_RETURN_NONE;
}

typedef struct
{
    PyArrayObject base;
    bh_base *ary;
    PyObject *bhc_ary;
}BhArray;

static PyObject *
BhArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    bh_error err;
    if(dtype_npy2bh == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Must call dtype_set_map() once, "
                                         "before creating new arrays");
        return NULL;
    }

    //First we lets NumPy create the base ndarray
    BhArray *self = (BhArray*) PyArray_Type.tp_new(type, args, kwds);

    //Convert data type to Bohrium
    bh_type t = type_py2cph(PyArray_TYPE(&self->base));
    if(t == BH_UNKNOWN)
    {
        PyErr_Format(PyExc_TypeError, "The dtype %s is not supported by Bohrium",
                     bh_npy_type_text(PyArray_TYPE(&self->base)));
        return NULL;
    }

//#    printf("BhArray_new(dtype: %s)\n", bh_type_text(t));
    err = bh_create_base(t, PyArray_SIZE((PyArrayObject*)self), &self->ary);
    if(err != BH_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Couldn't create new Bohrium array: %s",
                     bh_error_text(err));
        return NULL;

    }

    PyObject *m = PyImport_ImportModule("_util");
    if(m == NULL)
        return NULL;

    self->bhc_ary = PyObject_CallMethod(m, "create_bhc_array", "O", (PyObject *)self);
    if(self->bhc_ary == NULL)
        return NULL;
    Py_DECREF(m);

    return (PyObject *)self;
}

static void
BhArray_dealloc(BhArray* self)
{
    Py_XDECREF(self->bhc_ary);
}

static PyObject *
BhArray_get_bhc_ary(BhArray *self, void *closure)
{
    Py_INCREF(self->bhc_ary);
    return self->bhc_ary;
}
static PyGetSetDef BhArray_getseters[] = {
    {"bhc_ary",
     (getter)BhArray_get_bhc_ary,
     (setter)NULL,
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
    0,                       /* tp_methods */
    0,                       /* tp_members */
    BhArray_getseters,       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    0,                       /* tp_init */
    0,                       /* tp_alloc */
    (newfunc)BhArray_new,    /* tp_new */
};

static PyObject *
bh_exec_instr(PyObject *self, PyObject *args)
{
    PyObject *instr_dict;
    PyObject *ops;
//    Py_ssize_t i;


    if(!PyArg_ParseTuple(args, "OO", &instr_dict, &ops))
        return NULL;

    if(PyDict_Check(instr_dict) || !PySequence_Check(ops))
    {
        PyErr_Format(PyExc_TypeError, "The first argument 'instr_dict' must "
                "be a dict describing the operation and the second argument "
                "'operands' must be a list of bohrium arrays");
        return NULL;
    }
/*
    bh_instruction instr;

    for(i=0; i<PySequence_Size(ops); ++i)
    {
        PyObject *o = PySequence_GetItem(ops,i);
        if(o == NULL)
            return NULL;
        if(!BhArray_CheckExact(o))
        {
            PyErr_Format(PyExc_TypeError, "The operands must be bohrium arrays");
            return NULL;
        }
        bh_ir bhir;
        bh_error err = bh_ir_create(&bhir, 1, &instr);
    }
*/
    Py_RETURN_NONE;
}

static PyMethodDef BohriumMethods[] = {
    {"dtype_set_map", dtype_set_map, METH_VARARGS,
     "Set the data type map (dict) from NumPy to Bohrium (e.g. NPY_FLOAT32 to BH_FLOAT32)"},
    {"exec_instr", bh_exec_instr, METH_VARARGS,
     "Execute a instruction(instr_dict, args)"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
init_bh(void)
{
    PyObject *m;

    m = Py_InitModule("_bh", BohriumMethods);
    if (m == NULL)
        return;

    bhc_handle = dlopen("/home/madsbk/repos/bohrium/bridge/c/libbhc.so", RTLD_NOW);

    if(bhc_handle == NULL)
    {
        PyErr_Format(PyExc_ImportError, "Could not find the Bohrium C-Bridge "
                                        "library (%s)", dlerror());
        return;
    }

    //Import NumPy
    import_array();

    BhArrayType.tp_base = &PyArray_Type;
    if (PyType_Ready(&BhArrayType) < 0)
        return;

    PyModule_AddObject(m, "ndarray", (PyObject *)&BhArrayType);
}

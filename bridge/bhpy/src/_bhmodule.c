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
#include <bh_c.h>
#include <bh.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define BhArray_CheckExact(op) (((PyObject*)(op))->ob_type == &BhArrayType)


typedef struct
{
    PyArrayObject base;
    bh_base *ary;
}BhArray;

static PyTypeObject BhArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                       /* ob_size */
    "bohrium.ndarray",       /* tp_name */
    sizeof(BhArray),         /* tp_basicsize */
    0,                       /* tp_itemsize */
    0,                       /* tp_dealloc */
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
    0,                       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    0,                       /* tp_init */
    0,                       /* tp_alloc */
    0,                       /* tp_new */
};

static void *bhc_handle;

static PyObject *
bh_exec(PyObject *self, PyObject *args)
{
    const char *name;
    PyObject *ops;
    Py_ssize_t i;

    if(!PyArg_ParseTuple(args, "sO", &name, &ops))
        return NULL;

    if(PyString_Check(ops) || !PySequence_Check(ops))
    {
        PyErr_Format(PyExc_TypeError, "The first argument 'name' must be a string and "
                "the second argument 'operands' must be a list of bohrium arrays");
        return NULL;
    }

    for(i=0; i<PySequence_Size(ops); ++i)
    {
        PyObject *o = PySequence_GetItem(ops,i);
        if(o == NULL)
            return NULL;
        if(!BhArray_CheckExact(o))
        {
            PyErr_Format(PyExc_TypeError, "The operands must bohrium arrays");
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyMethodDef BohriumMethods[] = {
    {"execute", bh_exec, METH_VARARGS,
     "Execute function exec(name, args)"},
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

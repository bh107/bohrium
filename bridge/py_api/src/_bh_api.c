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

#if PY_MAJOR_VERSION >= 3
#define NPY_PY3K
#endif

#define BhAPI_MODULE

#include "bohrium_api.h"
#include <bhc.h>


static BhAPI_flush_RETURN BhAPI_flush BhAPI_flush_PROTO {
    bhc_flush();
};


PyObject *PyFlush(PyObject *self, PyObject *args) {
    printf("PyFlush()\n");
    BhAPI_flush();
    Py_RETURN_NONE;
}

PyObject *PyMessage(PyObject *self, PyObject *args, PyObject *kwds) {
    char *msg;
    static char *kwlist[] = {"msg:str", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &msg)) {
        return NULL;
    }
#if defined(NPY_PY3K)
    return PyUnicode_FromString(bhc_message(msg));
#else
    return PyString_FromString(bhc_message(msg));
#endif
}

// The methods (functions) of this module
static PyMethodDef _bh_apiMethods[] = {
        {"flush",   PyFlush,                 METH_NOARGS, "Evaluate all delayed array operations"},
        {"message", (PyCFunction) PyMessage, METH_VARARGS | METH_KEYWORDS,
                                                          "Send and receive a message through the Bohrium stack\n"},
        {NULL,      NULL,                    0,           NULL}        /* Sentinel */
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bh_api",/* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module or -1 if the module keeps state in global variables. */
    _bh_apiMethods /* the methods of this module */
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__bh_api(void)
#else
#define RETVAL

PyMODINIT_FUNC init_bh_api(void)
#endif
{
    static void *PyBhAPI[BhAPI_num_of_pointers];
    PyObject *c_api_object;
    PyObject *m;
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_bh_api", _bh_apiMethods);
#endif
    if (m == NULL) {
        return RETVAL;
    }

    /* Initialize the C API pointer array */
    PyBhAPI[BhAPI_flush_NUM] = (void *)BhAPI_flush;

    /* Create a Capsule containing the API pointer array's address */
    c_api_object = PyCapsule_New((void *)PyBhAPI, "bohrium_api._C_API", NULL);
    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    return RETVAL;
}

#include "Python.h"
#include "numpy/noprefix.h"
#include "structmember.h"
#include "numpy/arrayobject.h"

static PyObject *
distio_save(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyArrayObject *ary;
    char *filename;
    long datapos;

    if(!PyArg_ParseTuple(args, "O&sl", PyArray_Converter, &ary, &filename, &datapos))
        return NULL;

    /* Call DistNumPy */
    dnumpy_datadump(PyArray_DNDUID(ary), filename, datapos);

    Py_XDECREF(ary);

    return Py_None;
}

static PyObject *
distio_load(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"filename","datapos","shape","fortran_order","dtype",NULL};
    char *filename;
    long datapos;
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    int fortran_order = 0;
    PyObject *ret = NULL;
    int flags = DNPY_DISTRIBUTED;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "slO&iO&",
                                     kwlist, &filename, &datapos,
                                     PyArray_IntpConverter,
                                     &shape,
                                     &fortran_order,
                                     PyArray_DescrConverter,
                                     &typecode)) {
        goto fail;
    }

    if (fortran_order) {
        flags |= PyArray_FORTRANORDER;
    } else {
        flags |= PyArray_CORDER;
    }

    if (!typecode) {
        typecode = PyArray_DescrFromType(PyArray_DEFAULT);
    }

    /* Create the distributed array */
    ret = (PyObject *)PyArray_NewFromDescr(&PyArray_Type,
                                            typecode,
                                            shape.len, shape.ptr,
                                            NULL, NULL,
                                            flags, NULL);

    if (ret == NULL) {
        return NULL;
    }

    /* Call DistNumPy */
    dnumpy_datafill(PyArray_DNDUID(ret), filename, datapos);

    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return ret;
}


/* Export methods */
static struct PyMethodDef methods[] = {
    {"dist_load",  (PyCFunction)distio_load, METH_VARARGS | METH_KEYWORDS,
     "Load binary file into distributed array."},
    {"dist_save",  (PyCFunction)distio_save, METH_VARARGS,
     "Save distributed array into binary file."},
    {NULL, NULL}    /* sentinel */
};


/* Initialization function for the module (*must* be called init<name>) */

PyMODINIT_FUNC init_distio(void) {
    PyObject *m;

    /* Create the module and add the functions */
    m = Py_InitModule("_distio", methods);

    import_array();

    return;
}

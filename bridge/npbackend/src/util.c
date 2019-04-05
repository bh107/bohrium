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

#include <strings.h>
#include "util.h"
#include "bharray.h"

bhc_dtype dtype_np2bhc(const int np_dtype_num) {
    switch(np_dtype_num) {
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


int normalize_operand(PyObject *op, bhc_dtype *dtype, bhc_bool *constant, void **operand,
                      normalize_cleanup_handle *cleanup) {
    if (IsAnyScalar(op)) {
        // Convert any kind of scalar to a 0-dim array, which makes it easy to extract the scalar value
        PyObject *zero_dim_ary = PyArray_FromAny(op, NULL, 0, 1, 0, NULL);
        if (zero_dim_ary == NULL) {
            return -1;
        }
        *dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) zero_dim_ary)->type_num);
        *operand = PyArray_DATA((PyArrayObject*) zero_dim_ary);
        cleanup->objs2free[cleanup->objs2free_count++] = zero_dim_ary;
        *constant = 1;
    } else {
        // Let's make sure that we have a BhArray
        PyObject *bh_ary;
        if (BhArray_CheckExact(op)) {
            bh_ary = op;
        } else {
            bh_ary = PyObject_CallMethod(array_create, "array", "O", op);
            if(bh_ary == NULL) {
                return -1;
            }
            cleanup->objs2free[cleanup->objs2free_count++] = bh_ary;
        }
        assert(BhArray_CheckExact(bh_ary));

        // A zero sized view, we can ignore the whole operation
        if (PyArray_SIZE((PyArrayObject*) bh_ary) <= 0) {
            return -1;
        }

        *dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) bh_ary)->type_num);
        *constant = 0;
        *operand = bharray_bhc((BhArray*) bh_ary);
    }
    return 0;
}

void normalize_operand_cleanup(normalize_cleanup_handle *cleanup) {
    for (int i = 0; i < cleanup->objs2free_count; i++) {
        assert(cleanup->objs2free[i] != NULL);
        Py_DECREF(cleanup->objs2free[i]);
    }
    cleanup->objs2free_count = 0;
}

int64_t ary_nbytes(const BhArray *ary) {
    int64_t size = PyArray_NBYTES((PyArrayObject*) ary);
    if(size == 0) {
        return PyArray_ITEMSIZE((PyArrayObject*) ary);
    } else {
        return size;
    }
}

BhArray *get_base(PyObject *ary) {
    PyObject *base;
    if(PyArray_Check(ary)) {
        base = PyArray_BASE((PyArrayObject*) ary);
    } else {
        base = PyObject_GetAttrString(ary, "base");
        if(base == NULL) {
            PyErr_Format(PyExc_ValueError, "get_base() - the object has no base!\n");
            return NULL;
        }
        Py_DECREF(base); // Notice, we are returning a borrowed reference
    }
    if (base == NULL || base == Py_None) {
        if(!BhArray_CheckExact(ary)) {
            PyErr_Format(PyExc_ValueError, "get_base() -  the base object isn't a bohrium array!\n");
            return NULL;
        }
        return (BhArray *) ary;
    } else {
        return get_base(base);
    }
}

int same_view(PyArrayObject *v1, PyArrayObject *v2) {
    if (PyArray_TYPE(v1) != PyArray_TYPE(v2)) {
        return 0;
    }
    if (PyArray_DATA(v1) != PyArray_DATA(v2)) {
        return 0;
    }
    if ((PyArray_NDIM(v1) == 0 || PyArray_SIZE(v1) == 1) && (PyArray_NDIM(v2) == 0 || PyArray_SIZE(v2) == 1)) {
        return 1; // single element views are identical
    }
    if (PyArray_NDIM(v1) != PyArray_NDIM(v2)) {
        return 0;
    }
    for(int i=0; i < PyArray_NDIM(v1); ++i) {
        if (PyArray_DIM(v1, i) != PyArray_DIM(v2, i)) {
            return 0;
        }
        if (PyArray_STRIDE(v1, i) != PyArray_STRIDE(v2, i)) {
            return 0;
        }
    }
    return 1;
}

PyObject *PySameView(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *v1, *v2;
    static char *kwlist[] = {"v1:ndarray", "v2:ndarray", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &v1, &v2)) {
        return NULL;
    }
    if (!PyArray_Check(v1) || !PyArray_Check(v2)) {
        PyErr_Format(PyExc_TypeError, "The views must be a ndarray or a subtype thereof.");
        return NULL;
    }
    if (same_view((PyArrayObject*) v1, (PyArrayObject*) v2)) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

PyObject *PyIsBehaving(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary;
    static char *kwlist[] = {"ary:bharray", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &ary)) {
        return NULL;
    }
    if (!BhArray_CheckExact(ary)) {
        PyErr_Format(PyExc_TypeError, "Array must be a Bohrium array.");
        return NULL;
    }
    BhArray *base = get_base(ary);
    if (base == NULL) {
        return NULL;
    }
    if (PyArray_DATA(ary) != PyArray_DATA(base)) {
        Py_RETURN_FALSE; // `ary` uses an offset
    }
    if (!PyArray_IS_C_CONTIGUOUS(ary)) {
        Py_RETURN_FALSE; // `ary` is not C-style contiguous
    }
    if (PyArray_SIZE(ary) != PyArray_SIZE(base)) {
        Py_RETURN_FALSE; // `ary` does not represent the whole of its base
    }
    Py_RETURN_TRUE;
}

int get_bool_env(const char *name, int default_value) {
    const char *value = getenv(name);
    if (value != NULL) {
        const int slen = strlen(value);
        if (slen == 1) {
            if (strcasecmp(value, "1") == 0 || strcasecmp(value, "y") == 0 || strcasecmp(value, "t") == 0) {
                return 1;
            } else if (strcasecmp(value, "0") == 0 || strcasecmp(value, "n") == 0 || strcasecmp(value, "f") == 0) {
                return 0;
            }
        } else if (slen == 4 && strcasecmp(value, "true") == 0) {
            return 1;
        } else if (slen == 5 && strcasecmp(value, "false") == 0) {
            return 0;
        }
        fprintf(stderr, "Warning: \"%s=%s\" must be a boolean value, using default value: `%d`\n", name, value, default_value);
    }
    return default_value;
}

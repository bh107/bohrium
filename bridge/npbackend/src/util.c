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

#include "util.h"

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

        *dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) bh_ary)->type_num);
        *constant = 0;

        // Get the bhc array pointer and save it in `operands[i]`
        PyObject *bhc_view = PyObject_CallMethod(bhary, "get_bhc", "O", bh_ary);
        if(bhc_view == NULL) {
            return -1;
        }
        cleanup->objs2free[cleanup->objs2free_count++] = bhc_view;

        // A zero sized view has no `bhc_obj` and we can simply ignore the operation
        if(!PyObject_HasAttrString(bhc_view, "bhc_obj")) {
            return -1;
        }

        PyObject *bhc_ary_swig_ptr = PyObject_GetAttrString(bhc_view, "bhc_obj");
        if(bhc_ary_swig_ptr == NULL) {
            return -1;
        }
        PyObject *bhc_ary_ptr = PyObject_CallMethod(bhc_ary_swig_ptr, "__int__", NULL);
        if(bhc_ary_ptr == NULL) {
            Py_DECREF(bhc_ary_swig_ptr);
            return -1;
        }
        *operand = PyLong_AsVoidPtr(bhc_ary_ptr);
        Py_DECREF(bhc_ary_swig_ptr);
        Py_DECREF(bhc_ary_ptr);
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
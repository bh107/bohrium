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

#include <bhc.h>
#include "ufunc.h"

PyObject *
PyUfunc(PyObject *self, PyObject *args, PyObject *kwds) {
    int opcode;
    PyObject *operand_fast_seq;
    {
        PyObject *operand_list;
        static char *kwlist[] = {"opcode", "operand_list:list", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO", kwlist, &opcode, &operand_list)) {
            return NULL;
        }
        operand_fast_seq = PySequence_Fast(operand_list, "`operand_list` should be a sequence.");
        if (operand_fast_seq == NULL) {
            return NULL;
        }
    }

    const Py_ssize_t nop = PySequence_Fast_GET_SIZE(operand_fast_seq);
    bhc_dtype types[nop];
    bhc_bool constants[nop];
    void *operands[nop];
    // We have to delay some clean up to after the bhc API call.
    // `nop*3` is the maximum number of delayed frees, which should be more than enough.
    PyObject *objs2free[nop*3];
    int objs2free_count = 0;
    for (int i = 0; i < nop; ++i) {
        PyObject *op = PySequence_Fast_GET_ITEM(operand_fast_seq, i); // Borrowed reference and will not fail
        if (IsAnyScalar(op)) {
            // Convert any kind of scalar to a 0-dim array, which makes it easy to extract the scalar value
            PyObject *zero_dim_ary = PyArray_FromAny(op, NULL, 0, 1, 0, NULL);
            if (zero_dim_ary == NULL) {
                return NULL;
            }
            types[i] = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) zero_dim_ary)->type_num);
            operands[i] = PyArray_DATA((PyArrayObject*) zero_dim_ary);
            objs2free[objs2free_count++] = zero_dim_ary;
            constants[i] = 1;
        } else {
            // Let's make sure that we have a BhArray
            PyObject *bh_ary;
            if (BhArray_CheckExact(op)) {
                bh_ary = op;
            } else {
                bh_ary = PyObject_CallMethod(array_create, "array", "O", op);
                if(bh_ary == NULL) {
                    return NULL;
                }
                objs2free[objs2free_count++] = bh_ary;
            }
            assert(BhArray_CheckExact(bh_ary));

            types[i] = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) bh_ary)->type_num);
            constants[i] = 0;

            // Get the bhc array pointer and save it in `operands[i]`
            PyObject *bhc_view = PyObject_CallMethod(bhary, "get_bhc", "O", bh_ary);
            if(bhc_view == NULL) {
                return NULL;
            }
            objs2free[objs2free_count++] = bhc_view;

            // A zero sized view has no `bhc_obj` and we can simply ignore the operation
            if(!PyObject_HasAttrString(bhc_view, "bhc_obj")) {
                // Clean up
                for (int j = 0; j < objs2free_count; j++) {
                    assert(objs2free[j] != NULL);
                    Py_DECREF(objs2free[j]);
                }
                Py_DECREF(operand_fast_seq);
                Py_RETURN_NONE;
            }

            PyObject *bhc_ary_swig_ptr = PyObject_GetAttrString(bhc_view, "bhc_obj");
            if(bhc_ary_swig_ptr == NULL) {
                return NULL;
            }
            PyObject *bhc_ary_ptr = PyObject_CallMethod(bhc_ary_swig_ptr, "__int__", NULL);
            if(bhc_ary_ptr == NULL) {
                return NULL;
            }
            operands[i] = PyLong_AsVoidPtr(bhc_ary_ptr);
            Py_XDECREF(bhc_ary_swig_ptr);
            Py_XDECREF(bhc_ary_ptr);
        }
    }

    bhc_op(opcode, types, constants, operands);

    // Clean up
    for (int i = 0; i < nop; i++) {
        assert(objs2free[i] != NULL);
        Py_DECREF(objs2free[i]);
    }
    Py_DECREF(operand_fast_seq);
    Py_RETURN_NONE;
}
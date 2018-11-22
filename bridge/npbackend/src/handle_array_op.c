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

#include "handle_array_op.h"

PyObject *array_op(int opcode, const Py_ssize_t nop, PyObject **operand_list) {
    // Read and normalize all operands
    bhc_dtype types[nop];
    bhc_bool constants[nop];
    void *operands[nop];
    normalize_cleanup_handle cleanup;
    cleanup.objs2free_count = 0;
    for (int i = 0; i < nop; ++i) {
        PyObject *op = operand_list[i];

        // We have to handle float view of complex bases, which is something NumPy supports
        if(!IsAnyScalar(op) && PyArray_Check(op)) {
            BhArray *base = get_base(op);
            if (base == NULL) {
                normalize_operand_cleanup(&cleanup);
                return NULL;
            }
            if(PyArray_ISFLOAT((PyArrayObject *) op) && PyArray_ISCOMPLEX((PyArrayObject *) base)) {
                if (i == 0) {
                    PyErr_Format(PyExc_ValueError, "Sorry - Bohrium does't handle writing to "
                            "the imag or real part of a complex array. Will be fixed ASAP!\n");
                    normalize_operand_cleanup(&cleanup);
                    return NULL;
                }

                // All this is simply a hack to reinterpret 'op' as a complex view of the 'base'
                size_t v = (size_t) PyArray_DATA((PyArrayObject *) op);
                size_t b = (size_t) PyArray_DATA((PyArrayObject *) base);
                size_t offset = (v - b) / PyArray_ITEMSIZE((PyArrayObject *) base);
                void *tmp_data = PyArray_BYTES((PyArrayObject *) base) +
                                 offset * PyArray_ITEMSIZE((PyArrayObject *) base);
                PyObject *tmp_ary = PyArray_New(&BhArrayType,
                                                PyArray_NDIM((PyArrayObject *) op),
                                                PyArray_DIMS((PyArrayObject *) op),
                                                PyArray_TYPE((PyArrayObject *) base),
                                                PyArray_STRIDES((PyArrayObject *) op),
                                                tmp_data,
                                                PyArray_ITEMSIZE((PyArrayObject *) base),
                                                0, NULL);
                if(tmp_ary == NULL) {
                    normalize_operand_cleanup(&cleanup);
                    return NULL;
                }
                Py_INCREF(base);
                PyArray_SetBaseObject((PyArrayObject *) tmp_ary, (PyObject*) base);
                cleanup.objs2free[cleanup.objs2free_count++] = tmp_ary;
                // At this point `tmp_ary` is a regular view of the complex base
                // We can now copy `tmp_ary` into a new float array using either `BHC_REAL` or `BHC_IMAG`
                op = PyArray_New(&BhArrayType,
                                 PyArray_NDIM((PyArrayObject *) tmp_ary),
                                 PyArray_DIMS((PyArrayObject *) tmp_ary),
                                 PyArray_TYPE((PyArrayObject *) op),
                                 NULL, NULL,
                                 PyArray_ITEMSIZE((PyArrayObject *) op),
                                 0, NULL);
                if(op == NULL) {
                    normalize_operand_cleanup(&cleanup);
                    return NULL;
                }
                cleanup.objs2free[cleanup.objs2free_count++] = op;
                PyObject *tmp_operands[2] = {op, tmp_ary};
                int tmp_opcode;
                if ((v - b) % PyArray_ITEMSIZE((PyArrayObject *) base) == 0) {
                    tmp_opcode = BHC_REAL;
                } else {
                    tmp_opcode = BHC_IMAG;
                }
                if (array_op(tmp_opcode, 2, tmp_operands) == NULL) {
                    cleanup.objs2free[cleanup.objs2free_count++] = op;
                    return NULL;
                }
            }
        }

        int err = normalize_operand(op, &types[i], &constants[i], &operands[i], &cleanup);
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                Py_RETURN_NONE;
            }
        }
    }

    BhAPI_op(opcode, types, constants, operands);

    // Clean up
    normalize_operand_cleanup(&cleanup);
    Py_RETURN_NONE;
}

PyObject *
PyArrayOp(PyObject *self, PyObject *args, PyObject *kwds) {
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
    PyObject *ret = array_op(opcode, PySequence_Fast_GET_SIZE(operand_fast_seq),
                             PySequence_Fast_ITEMS(operand_fast_seq));
    Py_DECREF(operand_fast_seq);
    return ret;
}
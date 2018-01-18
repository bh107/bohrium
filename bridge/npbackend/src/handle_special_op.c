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
#include "util.h"

PyObject *
PyExtMethod(PyObject *self, PyObject *args, PyObject *kwds) {
    char *name;
    PyObject *operand_fast_seq;
    {
        PyObject *operand_list;
        static char *kwlist[] = {"name", "operand_list:list", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", kwlist, &name, &operand_list)) {
            return NULL;
        }
        operand_fast_seq = PySequence_Fast(operand_list, "`operand_list` should be a sequence.");
        if (operand_fast_seq == NULL) {
            return NULL;
        }
    }
    const Py_ssize_t nop = PySequence_Fast_GET_SIZE(operand_fast_seq);
    if (nop != 3) {
        PyErr_Format(PyExc_TypeError, "Expects three operands.");
        Py_DECREF(operand_fast_seq);
        return NULL;
    }

    // Read and normalize all operands
    bhc_dtype types[nop];
    bhc_bool constant;
    void *operands[nop];
    normalize_cleanup_handle cleanup;
    cleanup.objs2free_count = 0;
    for (int i = 0; i < nop; ++i) {
        PyObject *op = PySequence_Fast_GET_ITEM(operand_fast_seq, i); // Borrowed reference and will not fail
        int err = normalize_operand(op, &types[i], &constant, &operands[i], &cleanup);
        if (err != -1) {
            if (constant) {
                PyErr_Format(PyExc_TypeError, "Scalars isn't supported.");
                err = -1;
            } else if (types[0] != types[i]) {
                PyErr_Format(PyExc_TypeError, "The dtype of all operands must be the same.");
                err = -1;
            }
        }
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            Py_DECREF(operand_fast_seq);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                Py_RETURN_NONE;
            }
        }

    }

    int err = bhc_extmethod(types[0], name, operands[0], operands[1], operands[2]);
    if (err) {
        PyErr_Format(PyExc_TypeError, "The current runtime system does not support the extension method '%s'", name);
    }

    // Clean up
    normalize_operand_cleanup(&cleanup);
    Py_DECREF(operand_fast_seq);

    if (PyErr_Occurred() != NULL) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}
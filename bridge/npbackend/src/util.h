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

#pragma once

#include <bhc.h>
#define NO_IMPORT_ARRAY
#include "_bh.h"

/** This corresponds to `numpy.isscalar()`, which does not count 0-dim arrays as scalars
    In Bohrium, we handle 0-dim arrays as regular arrays. */
#define IsAnyScalar(o) (PyArray_IsScalar(o, Generic) || PyArray_IsPythonNumber(o))

/** Converts the dtype enum from NumPy to Bohrium */
bhc_dtype dtype_np2bhc(const int np_dtype_num);

/** Handle for normalize clean up*/
typedef struct _normalize_cleanup_handle {
    PyObject *objs2free[64];
    int objs2free_count;
} normalize_cleanup_handle;

/** Normalize the operand `op` and extract the dtype, the constant flag, and the operand bhc pointer.
 * NB: remember to call `normalize_operand_cleanup()` after you have used the extracted values
 *
 * @param op       The Python object to normalize and extract from. Can be a Numpy-array, Bohrium-array, and Scalar
 * @param dtype    The returned list of data types bhc enums
 * @param constant The returned list of booleans the indicate whether an operand is a constant or not
 * @param operand  The returned list of bhc array pointers and constants
 * @param cleanup  The clean up handle, which should be initiated with `objs2free_count = 0`
 * @return         Is 0 on success and -1 on abort. Use `PyErr_Occurred()` to check if its a Python error
 *                 or legit abort, in which case you can return Py_RETURN_NONE
 */
int normalize_operand(PyObject *op, bhc_dtype *dtype, bhc_bool *constant, void **operand,
                      normalize_cleanup_handle *cleanup);

/** Clean up after normalize_operand
 *  NB: `64` is the maximum number of delayed frees, which should be more than enough.
 *
 * @param cleanup  The clean up handle
 */
void normalize_operand_cleanup(normalize_cleanup_handle *cleanup);
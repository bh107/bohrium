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

#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_BH_API
#include "_bh.h"

/** This corresponds to `numpy.isscalar()`, which does not count 0-dim arrays as scalars
    In Bohrium, we handle 0-dim arrays as regular arrays. */
#define IsAnyScalar(o) (PyArray_IsScalar(o, Generic) || PyArray_IsPythonNumber(o))

/** Check if Python object is a BhArray */
#define BhArray_CheckExact(op) (((PyObject*) (op))->ob_type == &BhArrayType)

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
 * @param dtype    The returned data types bhc enums
 * @param constant The returned booleans the indicate whether an operand is a constant or not
 * @param operand  The returned bhc array pointers and constants
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

/** Returns number of bytes in 'ary' BUT minimum 'itemsize', which mimic the behavior of NumPy.
 *
 * @param ary  The array in question
 * @return     Number of bytes in `ary`
 */
int64_t ary_nbytes(const BhArray *ary);

/** Return the final base of `ary`. Return NULL and sets a Python exception on error.
 *  Notice, this function will only invoke Python when `ary` or one of it bases are not ndarray or bharrays.
 *
 * @param ary  The array in question
 * @return     A borrowed reference to the base of `ary`
 */
BhArray *get_base(PyObject *ary);

/** Return true when `v1` and `v2` is exactly the same (incl. pointing to the same base)
 *
 * @param v1  First array in question
 * @param v2  Second array in question
 * @return    The boolean answer
 */
int same_view(PyArrayObject *v1, PyArrayObject *v2);
PyObject *PySameView(PyObject *self, PyObject *args, PyObject *kwds);

/** Check if `ary` is a "behaving" Bohrium array, which requires:
 *    - C-style contiguous
 *    - Points to the first element in the underlying base array (no offset)
 *    - Has the same total length as its base
 *
 * @param ary The Bohrium array to check
 * @return The boolean answer
 */
PyObject *PyIsBehaving(PyObject *self, PyObject *args, PyObject *kwds);

/** Read an environment variable as a boolean value
 *
 * @param name           Name of the environment variable
 * @param default_value  Default value if environment variable isn't set
 * @return               Boolean value
 */
int get_bool_env(const char *name, int default_value);

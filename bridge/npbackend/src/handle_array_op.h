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

#include "util.h"

/** Handle regular array operations, which is the operations implemented in `bhc_array_operations_enum_typed.cpp`
 *
 * @param opcode        A enum opcode
 * @param nop           Number of operands
 * @param operand_list  Array of operands that can be NumPy-arrays, Bohrium-arrays, and Scalars
 *                      NB: the dtypes must match a bhc API function.
 */
PyObject *array_op(int opcode, const Py_ssize_t nop, PyObject **operand_list);

/** Handle regular array operations, which is the operations implemented in `bhc_array_operations_enum_typed.cpp`
 *  This function is similar to `array_op()` but handles Python arguments
 *
 * @param opcode        A enum opcode
 * @param operand_list  List of operands that can be NumPy-arrays, Bohrium-arrays, and Scalars
 *                      NB: the dtypes must match a bhc API function.
 */
PyObject * PyArrayOp(PyObject *self, PyObject *args, PyObject *kwds);
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

/** Handle extension methods
 *
 * @param name          The name of the extension method
 * @param operand_list  List of operands that can be NumPy-arrays and  Bohrium-arrays but NOT Scalars
 *                      NB: all dtype must be identical.
 */
PyObject *PyExtMethod(PyObject *self, PyObject *args, PyObject *kwds);


/** Execute the delayed instruction */
PyObject* PyFlush(PyObject *self, PyObject *args);

/** Sync `ary` to host memory */
PyObject* PySync(PyObject *self, PyObject *args, PyObject *kwds);

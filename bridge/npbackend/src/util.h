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
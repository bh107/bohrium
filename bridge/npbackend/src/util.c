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

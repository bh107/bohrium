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

#ifndef TYPES_H
#define TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

#include <bh.h>

//Check that the definitions in numpy are in accordance with Bohrium.
#if NPY_BITSOF_SHORT != 16
#    error the NPY_BITSOF_INT not 16 bit
#endif
#if NPY_BITSOF_INT != 32
#    error the NPY_BITSOF_INT not 32 bit
#endif
#if NPY_BITSOF_LONG != 32 && NPY_BITSOF_LONG != 64
#    error the NPY_BITSOF_LONG not 32 or 64 bit
#endif
#if NPY_BITSOF_LONGLONG != 64
#    error the NPY_BITSOF_LONGLONG not 64 bit
#endif
#if NPY_BITSOF_FLOAT != 32
#    error the NPY_BITSOF_FLOAT not 32 bit
#endif
#if NPY_BITSOF_FLOAT == 64
#    error the NPY_BITSOF_FLOAT not 64 bit
#endif


bh_type type_py2cph(int npy_type);
int type_cph2py(bh_type type);
bh_error bh_set_constant(int npy_type, bh_constant* constant, void * data);
bh_error bh_set_int_constant(int npy_type, bh_constant* constant,  long long integer);
const char* bh_npy_type_text(int npy_type);

#ifdef __cplusplus
}
#endif

#endif /* !defined(TYPES_H) */

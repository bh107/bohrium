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

#ifndef __OCLTYPE_HPP
#define __OCLTYPE_HPP

#include <bh_type.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    OCL_INT8,
    OCL_INT16,
    OCL_INT32,
    OCL_INT64,
    OCL_UINT8,
    OCL_UINT16,
    OCL_UINT32,
    OCL_UINT64,
    OCL_FLOAT32,
    OCL_FLOAT64,
    OCL_COMPLEX64,
    OCL_COMPLEX128,
    OCL_R123,
    OCL_UNKNOWN
} OCLtype;

OCLtype oclType(bh_type vbtype);
const char* oclTypeStr(OCLtype type);
const char* oclAPItypeStr(OCLtype type);
size_t oclSizeOf(OCLtype type);
bool isFloat(OCLtype type);
bool isComplex(OCLtype type);

#ifdef __cplusplus
}
#endif

#endif

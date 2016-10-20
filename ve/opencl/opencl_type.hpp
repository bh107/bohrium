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

#ifndef __OPENCL_DTYPE_HPP
#define __OPENCL_DTYPE_HPP

#include <cassert>

#include <bh_type.h>

// Return OpenCL API types, which are used inside the JIT kernels
const char* write_opencl_type(bh_type dtype)
{
    switch (dtype)
    {
        case BH_BOOL: return "uchar";
        case BH_INT8: return "char";
        case BH_INT16: return "short";
        case BH_INT32: return "int";
        case BH_INT64: return"long";
        case BH_UINT8: return "uchar";
        case BH_UINT16: return "ushort";
        case BH_UINT32: return "uint";
        case BH_UINT64: return "ulong";
        case BH_FLOAT32: return "float";
        case BH_FLOAT64: return "double";
        case BH_COMPLEX64: return "float2";
        case BH_COMPLEX128: return "double2";
        case BH_R123: return "ulong2";
        default:
            std::cerr << "Unknown OpenCL type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown OpenCL type");
    }
}

#endif

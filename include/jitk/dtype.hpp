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

#ifndef __BH_JITK_DTYPE_HPP
#define __BH_JITK_DTYPE_HPP

#include <sstream>
#include <stdexcept>

#include <bh_type.hpp>
#include <jitk/codegen_util.hpp>

namespace bohrium {
namespace jitk {

// Return C99 types, which are used inside the C99 kernels
const char *write_c99_type(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:       return "bool";
        case bh_type::INT8:       return "int8_t";
        case bh_type::INT16:      return "int16_t";
        case bh_type::INT32:      return "int32_t";
        case bh_type::INT64:      return "int64_t";
        case bh_type::UINT8:      return "uint8_t";
        case bh_type::UINT16:     return "uint16_t";
        case bh_type::UINT32:     return "uint32_t";
        case bh_type::UINT64:     return "uint64_t";
        case bh_type::FLOAT32:    return "float";
        case bh_type::FLOAT64:    return "double";
        case bh_type::COMPLEX64:  return "float complex";
        case bh_type::COMPLEX128: return "double complex";
        case bh_type::R123:       return "struct { uint64_t start, key; }";
        default:
            std::cerr << "Unknown C99 type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown C99 type");
    }
}

// Return OpenCL API types, which are used inside the JIT kernels
const char* write_opencl_type(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:       return "uchar";
        case bh_type::INT8:       return "char";
        case bh_type::INT16:      return "short";
        case bh_type::INT32:      return "int";
        case bh_type::INT64:      return "long";
        case bh_type::UINT8:      return "uchar";
        case bh_type::UINT16:     return "ushort";
        case bh_type::UINT32:     return "uint";
        case bh_type::UINT64:     return "ulong";
        case bh_type::FLOAT32:    return "float";
        case bh_type::FLOAT64:    return "double";
        case bh_type::COMPLEX64:  return "float2";
        case bh_type::COMPLEX128: return "double2";
        case bh_type::R123:       return "ulong2";
        default:
            std::cerr << "Unknown OpenCL type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown OpenCL type");
    }
}

// Return CUDA types, which are used inside the JIT kernels
const char *write_cuda_type(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:       return "bool";
        case bh_type::INT8:       return "char";
        case bh_type::INT16:      return "short";
        case bh_type::INT32:      return "int";
        case bh_type::INT64:      return "long";
        case bh_type::UINT8:      return "unsigned char";
        case bh_type::UINT16:     return "unsigned short";
        case bh_type::UINT32:     return "unsigned int";
        case bh_type::UINT64:     return "unsigned long";
        case bh_type::FLOAT32:    return "float";
        case bh_type::FLOAT64:    return "double";
        case bh_type::COMPLEX64:  return "cuFloatComplex";
        case bh_type::COMPLEX128: return "cuDoubleComplex";
        case bh_type::R123:       return "ulong2";
        default:
            std::cerr << "Unknown CUDA type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown CUDA type");
    }
}

// Writes the union of C99 types that can make up a constant
void write_c99_dtype_union(std::stringstream& out) {
    out << "union dtype {\n";
    spaces(out, 4);
    out << write_c99_type(bh_type::BOOL) << " " << bh_type_text(bh_type::BOOL) << ";\n";
    out << write_c99_type(bh_type::INT8) << " " << bh_type_text(bh_type::INT8) << ";\n";
    out << write_c99_type(bh_type::INT16) << " " << bh_type_text(bh_type::INT16) << ";\n";
    out << write_c99_type(bh_type::INT32) << " " << bh_type_text(bh_type::INT32) << ";\n";
    out << write_c99_type(bh_type::INT64) << " " << bh_type_text(bh_type::INT64) << ";\n";
    out << write_c99_type(bh_type::UINT8) << " " << bh_type_text(bh_type::UINT8) << ";\n";
    out << write_c99_type(bh_type::UINT16) << " " << bh_type_text(bh_type::UINT16) << ";\n";
    out << write_c99_type(bh_type::UINT32) << " " << bh_type_text(bh_type::UINT32) << ";\n";
    out << write_c99_type(bh_type::UINT64) << " " << bh_type_text(bh_type::UINT64) << ";\n";
    out << write_c99_type(bh_type::FLOAT32) << " " << bh_type_text(bh_type::FLOAT32) << ";\n";
    out << write_c99_type(bh_type::FLOAT64) << " " << bh_type_text(bh_type::FLOAT64) << ";\n";
    out << write_c99_type(bh_type::COMPLEX64) << " " << bh_type_text(bh_type::COMPLEX64) << ";\n";
    out << write_c99_type(bh_type::COMPLEX128) << " " << bh_type_text(bh_type::COMPLEX128) << ";\n";
    out << write_c99_type(bh_type::R123) << " " << bh_type_text(bh_type::R123) << ";\n";
    out << "};\n";
}

// Return Fortran types, which are used inside the Fortran kernels
const char *write_fortran_type(bh_type dtype) {
    switch (dtype) {
        case bh_type::BOOL:       return "logical";
        case bh_type::INT8:       return "integer*1";
        case bh_type::INT16:      return "integer*2";
        case bh_type::INT32:      return "integer*4";
        case bh_type::INT64:      return "integer*8";
        case bh_type::UINT8:      return "integer*1";
        case bh_type::UINT16:     return "integer*2";
        case bh_type::UINT32:     return "integer*4";
        case bh_type::UINT64:     return "integer*8";
        case bh_type::FLOAT32:    return "real*4";
        case bh_type::FLOAT64:    return "real*8";
        case bh_type::COMPLEX64:  return "complex*8";
        case bh_type::COMPLEX128: return "complex*16";
        case bh_type::R123:       return "struct { uint64_t start, key; }";
        default:
            std::cerr << "Unknown Fortran type: " << bh_type_text(dtype) << std::endl;
            throw std::runtime_error("Unknown Fortran type");
    }
}

} // jitk
} // bohrium

#endif

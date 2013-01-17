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
#include <bh.h>
#include <bh_compute.h>
#include "functors.hpp"
#include "traverser.hpp"
#include <complex>

bh_computeloop_naive bh_compute_get_naive( bh_instruction *instr ) {

    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
 
    if (bh_operands(instr->opcode) == 3) {   // Three operands

        if (bh_is_constant(instr->operand[1])) {             // First operand is constant
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->constant.type << 12) \
                +(instr->operand[2]->type << 16) \
                +(1 << 17);

        } else if (bh_is_constant(instr->operand[2])) {      // Second operand is constant
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12) \
                +(instr->constant.type << 16) \
                +(1 << 18);

        } else {                                                // No constant operand
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12) \
                +(instr->operand[2]->type << 16);
        }

    } else {                                    // Two operands

        if (bh_is_constant(instr->operand[1])) {
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->constant.type << 12) \
                +(1 << 17);
        } else {
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12);
        }
    }

    switch(poly) {

        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int8,bh_int8, absolute_functor<bh_int8,bh_int8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int8, absolute_functor<bh_int8,bh_int8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, absolute_functor<bh_float64,bh_float64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, absolute_functor<bh_float64,bh_float64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint16, absolute_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint16, absolute_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint64, absolute_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint64, absolute_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int16,bh_int16, absolute_functor<bh_int16,bh_int16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int16, absolute_functor<bh_int16,bh_int16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, absolute_functor<bh_float32,bh_float32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, absolute_functor<bh_float32,bh_float32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_bool,bh_bool, absolute_functor<bh_bool,bh_bool > >;
        case CPHVB_ABSOLUTE + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_bool, absolute_functor<bh_bool,bh_bool > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int32,bh_int32, absolute_functor<bh_int32,bh_int32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int32, absolute_functor<bh_int32,bh_int32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint8, absolute_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint8, absolute_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int64,bh_int64, absolute_functor<bh_int64,bh_int64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int64, absolute_functor<bh_int64,bh_int64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint32, absolute_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint32, absolute_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_naive_aaa<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_naive_aaa<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_bool,bh_int8, logical_not_functor<bh_bool,bh_int8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int8, logical_not_functor<bh_bool,bh_int8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_float64, logical_not_functor<bh_bool,bh_float64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float64, logical_not_functor<bh_bool,bh_float64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint16, logical_not_functor<bh_bool,bh_uint16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint16, logical_not_functor<bh_bool,bh_uint16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint64, logical_not_functor<bh_bool,bh_uint64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint64, logical_not_functor<bh_bool,bh_uint64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_bool,bh_int16, logical_not_functor<bh_bool,bh_int16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int16, logical_not_functor<bh_bool,bh_int16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_float32, logical_not_functor<bh_bool,bh_float32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float32, logical_not_functor<bh_bool,bh_float32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_bool,bh_bool, logical_not_functor<bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_bool, logical_not_functor<bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_int32, logical_not_functor<bh_bool,bh_int32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int32, logical_not_functor<bh_bool,bh_int32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint8, logical_not_functor<bh_bool,bh_uint8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint8, logical_not_functor<bh_bool,bh_uint8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_int64, logical_not_functor<bh_bool,bh_int64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int64, logical_not_functor<bh_bool,bh_int64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint32, logical_not_functor<bh_bool,bh_uint32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint32, logical_not_functor<bh_bool,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_naive_aaa<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_naive_aac<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_naive_aca<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_INVERT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int8,bh_int8, invert_functor<bh_int8,bh_int8 > >;
        case CPHVB_INVERT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int8, invert_functor<bh_int8,bh_int8 > >;
        case CPHVB_INVERT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint16, invert_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_INVERT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint16, invert_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_INVERT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint64, invert_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_INVERT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint64, invert_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_INVERT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int16,bh_int16, invert_functor<bh_int16,bh_int16 > >;
        case CPHVB_INVERT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int16, invert_functor<bh_int16,bh_int16 > >;
        case CPHVB_INVERT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_bool,bh_bool, invert_functor<bh_bool,bh_bool > >;
        case CPHVB_INVERT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_bool, invert_functor<bh_bool,bh_bool > >;
        case CPHVB_INVERT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int32,bh_int32, invert_functor<bh_int32,bh_int32 > >;
        case CPHVB_INVERT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int32, invert_functor<bh_int32,bh_int32 > >;
        case CPHVB_INVERT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint8, invert_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_INVERT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint8, invert_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_INVERT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int64,bh_int64, invert_functor<bh_int64,bh_int64 > >;
        case CPHVB_INVERT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int64, invert_functor<bh_int64,bh_int64 > >;
        case CPHVB_INVERT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint32, invert_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_INVERT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint32, invert_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_COS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, cos_functor<bh_float32,bh_float32 > >;
        case CPHVB_COS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, cos_functor<bh_float32,bh_float32 > >;
        case CPHVB_COS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, cos_functor<bh_float64,bh_float64 > >;
        case CPHVB_COS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, cos_functor<bh_float64,bh_float64 > >;
        case CPHVB_COS + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, cos_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COS + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, cos_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COS + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, cos_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COS + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, cos_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, sin_functor<bh_float32,bh_float32 > >;
        case CPHVB_SIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, sin_functor<bh_float32,bh_float32 > >;
        case CPHVB_SIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, sin_functor<bh_float64,bh_float64 > >;
        case CPHVB_SIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, sin_functor<bh_float64,bh_float64 > >;
        case CPHVB_SIN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, sin_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, sin_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, sin_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, sin_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, tan_functor<bh_float32,bh_float32 > >;
        case CPHVB_TAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, tan_functor<bh_float32,bh_float32 > >;
        case CPHVB_TAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, tan_functor<bh_float64,bh_float64 > >;
        case CPHVB_TAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, tan_functor<bh_float64,bh_float64 > >;
        case CPHVB_TAN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, tan_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, tan_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, tan_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, tan_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, cosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_COSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, cosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_COSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, cosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_COSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, cosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_COSH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, cosh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, cosh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, cosh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, cosh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, sinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_SINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, sinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_SINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, sinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_SINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, sinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_SINH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, sinh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, sinh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, sinh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, sinh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, tanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_TANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, tanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_TANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, tanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_TANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, tanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_TANH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, tanh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, tanh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, tanh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, tanh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arcsin_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arcsin_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arcsin_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arcsin_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arccos_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arccos_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arccos_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arccos_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arctan_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arctan_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arctan_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arctan_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arcsinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arcsinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arcsinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arcsinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arccosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arccosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arccosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arccosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, arctanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, arctanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, arctanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, arctanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, exp_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, exp_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, exp_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, exp_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, exp_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, exp_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, exp_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, exp_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, exp2_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, exp2_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, exp2_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, exp2_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, expm1_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, expm1_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, expm1_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, expm1_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, log_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, log_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, log_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, log_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, log_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, log_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, log_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, log_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, log2_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, log2_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, log2_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, log2_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, log10_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, log10_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, log10_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, log10_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, log10_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, log10_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, log10_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, log10_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, log1p_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, log1p_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, log1p_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, log1p_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, sqrt_functor<bh_float32,bh_float32 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, sqrt_functor<bh_float32,bh_float32 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, sqrt_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, sqrt_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, sqrt_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, sqrt_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, sqrt_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, sqrt_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_CEIL + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, ceil_functor<bh_float32,bh_float32 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, ceil_functor<bh_float32,bh_float32 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, ceil_functor<bh_float64,bh_float64 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, ceil_functor<bh_float64,bh_float64 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, trunc_functor<bh_float32,bh_float32 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, trunc_functor<bh_float32,bh_float32 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, trunc_functor<bh_float64,bh_float64 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, trunc_functor<bh_float64,bh_float64 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, floor_functor<bh_float32,bh_float32 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, floor_functor<bh_float32,bh_float32 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, floor_functor<bh_float64,bh_float64 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, floor_functor<bh_float64,bh_float64 > >;
        case CPHVB_RINT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, rint_functor<bh_float32,bh_float32 > >;
        case CPHVB_RINT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, rint_functor<bh_float32,bh_float32 > >;
        case CPHVB_RINT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, rint_functor<bh_float64,bh_float64 > >;
        case CPHVB_RINT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, rint_functor<bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_naive_aaa<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_naive_aaa<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_naive_aaa<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_naive_aaa<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_naive_aaa<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_naive_aaa<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_naive_aaa<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_naive_aaa<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_naive_aaa<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_naive_aaa<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_naive_aac<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_naive_aca<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_float32, isnan_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float32, isnan_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_float64, isnan_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float64, isnan_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_float32, isinf_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float32, isinf_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_float64, isinf_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float64, isinf_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_bool,bh_int8, identity_functor<bh_bool,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int8, identity_functor<bh_bool,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int8,bh_int8, identity_functor<bh_int8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int8, identity_functor<bh_int8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_uint8,bh_int8, identity_functor<bh_uint8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_int8, identity_functor<bh_uint8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int16,bh_int8, identity_functor<bh_int16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int8, identity_functor<bh_int16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_uint16,bh_int8, identity_functor<bh_uint16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_int8, identity_functor<bh_uint16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int32,bh_int8, identity_functor<bh_int32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int8, identity_functor<bh_int32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_uint32,bh_int8, identity_functor<bh_uint32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_int8, identity_functor<bh_uint32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_int64,bh_int8, identity_functor<bh_int64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int8, identity_functor<bh_int64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_uint64,bh_int8, identity_functor<bh_uint64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_int8, identity_functor<bh_uint64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_float32,bh_int8, identity_functor<bh_float32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_int8, identity_functor<bh_float32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<bh_float64,bh_int8, identity_functor<bh_float64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_int8, identity_functor<bh_float64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_float64, identity_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float64, identity_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_int8,bh_float64, identity_functor<bh_int8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_float64, identity_functor<bh_int8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_uint8,bh_float64, identity_functor<bh_uint8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_float64, identity_functor<bh_uint8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_int16,bh_float64, identity_functor<bh_int16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_float64, identity_functor<bh_int16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_uint16,bh_float64, identity_functor<bh_uint16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_float64, identity_functor<bh_uint16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_int32,bh_float64, identity_functor<bh_int32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_float64, identity_functor<bh_int32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_uint32,bh_float64, identity_functor<bh_uint32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_float64, identity_functor<bh_uint32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_int64,bh_float64, identity_functor<bh_int64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_float64, identity_functor<bh_int64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_uint64,bh_float64, identity_functor<bh_uint64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_float64, identity_functor<bh_uint64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float32,bh_float64, identity_functor<bh_float32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float64, identity_functor<bh_float32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_float64, identity_functor<bh_float64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float64, identity_functor<bh_float64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint16, identity_functor<bh_bool,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint16, identity_functor<bh_bool,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_int8,bh_uint16, identity_functor<bh_int8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_uint16, identity_functor<bh_int8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint16, identity_functor<bh_uint8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint16, identity_functor<bh_uint8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_int16,bh_uint16, identity_functor<bh_int16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_uint16, identity_functor<bh_int16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint16, identity_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint16, identity_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_int32,bh_uint16, identity_functor<bh_int32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_uint16, identity_functor<bh_int32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint16, identity_functor<bh_uint32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint16, identity_functor<bh_uint32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_int64,bh_uint16, identity_functor<bh_int64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_uint16, identity_functor<bh_int64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint16, identity_functor<bh_uint64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint16, identity_functor<bh_uint64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_float32,bh_uint16, identity_functor<bh_float32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_uint16, identity_functor<bh_float32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<bh_float64,bh_uint16, identity_functor<bh_float64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_uint16, identity_functor<bh_float64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint64, identity_functor<bh_bool,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint64, identity_functor<bh_bool,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_int8,bh_uint64, identity_functor<bh_int8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_uint64, identity_functor<bh_int8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint64, identity_functor<bh_uint8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint64, identity_functor<bh_uint8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_int16,bh_uint64, identity_functor<bh_int16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_uint64, identity_functor<bh_int16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint64, identity_functor<bh_uint16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint64, identity_functor<bh_uint16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_int32,bh_uint64, identity_functor<bh_int32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_uint64, identity_functor<bh_int32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint64, identity_functor<bh_uint32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint64, identity_functor<bh_uint32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_int64,bh_uint64, identity_functor<bh_int64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_uint64, identity_functor<bh_int64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint64, identity_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint64, identity_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_float32,bh_uint64, identity_functor<bh_float32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_uint64, identity_functor<bh_float32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_uint64, identity_functor<bh_float64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_uint64, identity_functor<bh_float64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_bool,bh_int16, identity_functor<bh_bool,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int16, identity_functor<bh_bool,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int8,bh_int16, identity_functor<bh_int8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int16, identity_functor<bh_int8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_uint8,bh_int16, identity_functor<bh_uint8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_int16, identity_functor<bh_uint8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int16,bh_int16, identity_functor<bh_int16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int16, identity_functor<bh_int16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_uint16,bh_int16, identity_functor<bh_uint16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_int16, identity_functor<bh_uint16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int32,bh_int16, identity_functor<bh_int32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int16, identity_functor<bh_int32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_uint32,bh_int16, identity_functor<bh_uint32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_int16, identity_functor<bh_uint32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_int64,bh_int16, identity_functor<bh_int64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int16, identity_functor<bh_int64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_uint64,bh_int16, identity_functor<bh_uint64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_int16, identity_functor<bh_uint64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_float32,bh_int16, identity_functor<bh_float32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_int16, identity_functor<bh_float32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<bh_float64,bh_int16, identity_functor<bh_float64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_int16, identity_functor<bh_float64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_float32, identity_functor<bh_bool,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_float32, identity_functor<bh_bool,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_int8,bh_float32, identity_functor<bh_int8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_float32, identity_functor<bh_int8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_uint8,bh_float32, identity_functor<bh_uint8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_float32, identity_functor<bh_uint8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_int16,bh_float32, identity_functor<bh_int16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_float32, identity_functor<bh_int16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_uint16,bh_float32, identity_functor<bh_uint16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_float32, identity_functor<bh_uint16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_int32,bh_float32, identity_functor<bh_int32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_float32, identity_functor<bh_int32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_uint32,bh_float32, identity_functor<bh_uint32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_float32, identity_functor<bh_uint32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_int64,bh_float32, identity_functor<bh_int64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_float32, identity_functor<bh_int64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_uint64,bh_float32, identity_functor<bh_uint64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_float32, identity_functor<bh_uint64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_float32, identity_functor<bh_float32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_float32, identity_functor<bh_float32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<bh_float64,bh_float32, identity_functor<bh_float64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_float32, identity_functor<bh_float64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_bool,bh_bool, identity_functor<bh_bool,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_bool, identity_functor<bh_bool,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_int8,bh_bool, identity_functor<bh_int8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_bool, identity_functor<bh_int8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_uint8,bh_bool, identity_functor<bh_uint8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_bool, identity_functor<bh_uint8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_int16,bh_bool, identity_functor<bh_int16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_bool, identity_functor<bh_int16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_uint16,bh_bool, identity_functor<bh_uint16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_bool, identity_functor<bh_uint16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_int32,bh_bool, identity_functor<bh_int32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_bool, identity_functor<bh_int32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_uint32,bh_bool, identity_functor<bh_uint32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_bool, identity_functor<bh_uint32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_int64,bh_bool, identity_functor<bh_int64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_bool, identity_functor<bh_int64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_uint64,bh_bool, identity_functor<bh_uint64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_bool, identity_functor<bh_uint64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_float32,bh_bool, identity_functor<bh_float32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_bool, identity_functor<bh_float32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<bh_float64,bh_bool, identity_functor<bh_float64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_bool, identity_functor<bh_float64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_int32, identity_functor<bh_bool,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int32, identity_functor<bh_bool,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int8,bh_int32, identity_functor<bh_int8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int32, identity_functor<bh_int8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_uint8,bh_int32, identity_functor<bh_uint8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_int32, identity_functor<bh_uint8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int16,bh_int32, identity_functor<bh_int16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int32, identity_functor<bh_int16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_uint16,bh_int32, identity_functor<bh_uint16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_int32, identity_functor<bh_uint16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int32,bh_int32, identity_functor<bh_int32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int32, identity_functor<bh_int32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_uint32,bh_int32, identity_functor<bh_uint32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_int32, identity_functor<bh_uint32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_int64,bh_int32, identity_functor<bh_int64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int32, identity_functor<bh_int64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_uint64,bh_int32, identity_functor<bh_uint64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_int32, identity_functor<bh_uint64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_int32, identity_functor<bh_float32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_int32, identity_functor<bh_float32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<bh_float64,bh_int32, identity_functor<bh_float64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_int32, identity_functor<bh_float64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint8, identity_functor<bh_bool,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint8, identity_functor<bh_bool,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_int8,bh_uint8, identity_functor<bh_int8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_uint8, identity_functor<bh_int8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint8, identity_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint8, identity_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_int16,bh_uint8, identity_functor<bh_int16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_uint8, identity_functor<bh_int16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint8, identity_functor<bh_uint16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint8, identity_functor<bh_uint16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_int32,bh_uint8, identity_functor<bh_int32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_uint8, identity_functor<bh_int32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint8, identity_functor<bh_uint32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint8, identity_functor<bh_uint32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_int64,bh_uint8, identity_functor<bh_int64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_uint8, identity_functor<bh_int64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint8, identity_functor<bh_uint64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint8, identity_functor<bh_uint64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_float32,bh_uint8, identity_functor<bh_float32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_uint8, identity_functor<bh_float32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<bh_float64,bh_uint8, identity_functor<bh_float64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_uint8, identity_functor<bh_float64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_bool,bh_int64, identity_functor<bh_bool,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_int64, identity_functor<bh_bool,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int8,bh_int64, identity_functor<bh_int8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_int64, identity_functor<bh_int8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_uint8,bh_int64, identity_functor<bh_uint8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_int64, identity_functor<bh_uint8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int16,bh_int64, identity_functor<bh_int16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_int64, identity_functor<bh_int16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_uint16,bh_int64, identity_functor<bh_uint16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_int64, identity_functor<bh_uint16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int32,bh_int64, identity_functor<bh_int32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_int64, identity_functor<bh_int32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_uint32,bh_int64, identity_functor<bh_uint32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_int64, identity_functor<bh_uint32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_int64,bh_int64, identity_functor<bh_int64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_int64, identity_functor<bh_int64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_uint64,bh_int64, identity_functor<bh_uint64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_int64, identity_functor<bh_uint64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_float32,bh_int64, identity_functor<bh_float32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_int64, identity_functor<bh_float32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<bh_float64,bh_int64, identity_functor<bh_float64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_int64, identity_functor<bh_float64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_bool,bh_uint32, identity_functor<bh_bool,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_bool,bh_uint32, identity_functor<bh_bool,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_int8,bh_uint32, identity_functor<bh_int8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int8,bh_uint32, identity_functor<bh_int8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint8,bh_uint32, identity_functor<bh_uint8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint8,bh_uint32, identity_functor<bh_uint8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_int16,bh_uint32, identity_functor<bh_int16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int16,bh_uint32, identity_functor<bh_int16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint16,bh_uint32, identity_functor<bh_uint16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint16,bh_uint32, identity_functor<bh_uint16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_int32,bh_uint32, identity_functor<bh_int32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int32,bh_uint32, identity_functor<bh_int32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint32,bh_uint32, identity_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint32,bh_uint32, identity_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_int64,bh_uint32, identity_functor<bh_int64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_int64,bh_uint32, identity_functor<bh_int64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_uint64,bh_uint32, identity_functor<bh_uint64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_uint64,bh_uint32, identity_functor<bh_uint64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_float32,bh_uint32, identity_functor<bh_float32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float32,bh_uint32, identity_functor<bh_float32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<bh_float64,bh_uint32, identity_functor<bh_float64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<bh_float64,bh_uint32, identity_functor<bh_float64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<float>, identity_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<float>, identity_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<double>, identity_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<double>, identity_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<std::complex<float>,bh_bool, identity_functor<std::complex<float>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_bool, identity_functor<std::complex<float>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_BOOL <<12):
            return &traverse_naive_aa<std::complex<double>,bh_bool, identity_functor<std::complex<double>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_bool, identity_functor<std::complex<double>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_int8, identity_functor<std::complex<float>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_int8, identity_functor<std::complex<float>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT8 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_int8, identity_functor<std::complex<double>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_int8, identity_functor<std::complex<double>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_uint8, identity_functor<std::complex<float>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_uint8, identity_functor<std::complex<float>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_uint8, identity_functor<std::complex<double>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_uint8, identity_functor<std::complex<double>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_int16, identity_functor<std::complex<float>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_int16, identity_functor<std::complex<float>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT16 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_int16, identity_functor<std::complex<double>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_int16, identity_functor<std::complex<double>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_uint16, identity_functor<std::complex<float>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_uint16, identity_functor<std::complex<float>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_uint16, identity_functor<std::complex<double>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_uint16, identity_functor<std::complex<double>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_int32, identity_functor<std::complex<float>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_int32, identity_functor<std::complex<float>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT32 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_int32, identity_functor<std::complex<double>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_int32, identity_functor<std::complex<double>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_uint32, identity_functor<std::complex<float>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_uint32, identity_functor<std::complex<float>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_uint32, identity_functor<std::complex<double>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_uint32, identity_functor<std::complex<double>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_int64, identity_functor<std::complex<float>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_int64, identity_functor<std::complex<float>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT64 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_int64, identity_functor<std::complex<double>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_int64, identity_functor<std::complex<double>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_uint64, identity_functor<std::complex<float>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_uint64, identity_functor<std::complex<float>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_uint64, identity_functor<std::complex<double>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_uint64, identity_functor<std::complex<double>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_float32, identity_functor<std::complex<float>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_float32, identity_functor<std::complex<float>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_float32, identity_functor<std::complex<double>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_float32, identity_functor<std::complex<double>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<std::complex<float>,bh_float64, identity_functor<std::complex<float>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,bh_float64, identity_functor<std::complex<float>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_naive_aa<std::complex<double>,bh_float64, identity_functor<std::complex<double>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,bh_float64, identity_functor<std::complex<double>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_naive_aa<std::complex<double>,std::complex<float>, identity_functor<std::complex<double>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<double>,std::complex<float>, identity_functor<std::complex<double>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_naive_aa<std::complex<float>,std::complex<double>, identity_functor<std::complex<float>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_naive_ac<std::complex<float>,std::complex<double>, identity_functor<std::complex<float>,std::complex<double> > >;

        default:
            return NULL;

    }

}

bh_computeloop bh_compute_get( bh_instruction *instr ) {

    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
 
    if (bh_operands(instr->opcode) == 3) {   // Three operands

        if (bh_is_constant(instr->operand[1])) {             // First operand is constant
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->constant.type << 12) \
                +(instr->operand[2]->type << 16) \
                +(1 << 17);

        } else if (bh_is_constant(instr->operand[2])) {      // Second operand is constant
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12) \
                +(instr->constant.type << 16) \
                +(1 << 18);

        } else {                                                // No constant operand
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12) \
                +(instr->operand[2]->type << 16);
        }

    } else {                                    // Two operands

        if (bh_is_constant(instr->operand[1])) {
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->constant.type << 12) \
                +(1 << 17);
        } else {
            poly = instr->opcode \
                +(instr->operand[0]->type << 8) \
                +(instr->operand[1]->type << 12);
        }
    }

    switch(poly) {

        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, add_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, add_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, add_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, add_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, add_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, add_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, add_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, add_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, add_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, add_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, add_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<std::complex<float>,std::complex<float>,std::complex<float>, add_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<std::complex<double>,std::complex<double>,std::complex<double>, add_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, subtract_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, subtract_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, subtract_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, subtract_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, subtract_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, subtract_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, subtract_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, subtract_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, subtract_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, subtract_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, subtract_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<std::complex<float>,std::complex<float>,std::complex<float>, subtract_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<std::complex<double>,std::complex<double>,std::complex<double>, subtract_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, multiply_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, multiply_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, multiply_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, multiply_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, multiply_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, multiply_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, multiply_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, multiply_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, multiply_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, multiply_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, multiply_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<std::complex<float>,std::complex<float>,std::complex<float>, multiply_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<std::complex<double>,std::complex<double>,std::complex<double>, multiply_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, divide_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, divide_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, divide_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, divide_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, divide_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, divide_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, divide_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, divide_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, divide_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, divide_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<std::complex<float>,std::complex<float>,std::complex<float>, divide_functor<std::complex<float>,std::complex<float>,std::complex<float> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<std::complex<double>,std::complex<double>,std::complex<double>, divide_functor<std::complex<double>,std::complex<double>,std::complex<double> > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, power_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, power_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, power_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, power_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, power_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, power_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, power_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, power_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, power_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_POWER + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, power_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int8,bh_int8, absolute_functor<bh_int8,bh_int8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int8, absolute_functor<bh_int8,bh_int8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, absolute_functor<bh_float64,bh_float64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, absolute_functor<bh_float64,bh_float64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint16,bh_uint16, absolute_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint16, absolute_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint64,bh_uint64, absolute_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint64, absolute_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int16,bh_int16, absolute_functor<bh_int16,bh_int16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int16, absolute_functor<bh_int16,bh_int16 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, absolute_functor<bh_float32,bh_float32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, absolute_functor<bh_float32,bh_float32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_bool,bh_bool, absolute_functor<bh_bool,bh_bool > >;
        case CPHVB_ABSOLUTE + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_bool, absolute_functor<bh_bool,bh_bool > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int32,bh_int32, absolute_functor<bh_int32,bh_int32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int32, absolute_functor<bh_int32,bh_int32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint8,bh_uint8, absolute_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint8, absolute_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int64,bh_int64, absolute_functor<bh_int64,bh_int64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int64, absolute_functor<bh_int64,bh_int64 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint32,bh_uint32, absolute_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_ABSOLUTE + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint32, absolute_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, greater_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, greater_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, greater_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, greater_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, greater_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, greater_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, greater_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, greater_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, greater_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, greater_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, greater_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, greater_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, greater_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, greater_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, greater_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, greater_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, greater_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, greater_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, greater_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, greater_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, greater_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, greater_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, less_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, less_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, less_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, less_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, less_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, less_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, less_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, less_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, less_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, less_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, less_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, less_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, less_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, less_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, less_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, less_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, less_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, less_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, less_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, less_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, less_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, less_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,std::complex<float>,std::complex<float>, equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,std::complex<double>,std::complex<double>, equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, not_equal_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, not_equal_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, not_equal_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, not_equal_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, not_equal_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, not_equal_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, not_equal_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, not_equal_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, not_equal_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, not_equal_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, not_equal_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16):
            return &traverse_aaa<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX64 <<12) + (CPHVB_COMPLEX64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,std::complex<float>,std::complex<float>, not_equal_functor<bh_bool,std::complex<float>,std::complex<float> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16):
            return &traverse_aaa<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8) +(CPHVB_COMPLEX128 <<12) + (CPHVB_COMPLEX128 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,std::complex<double>,std::complex<double>, not_equal_functor<bh_bool,std::complex<double>,std::complex<double> > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, logical_and_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, logical_and_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, logical_and_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, logical_and_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, logical_and_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, logical_and_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, logical_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, logical_and_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, logical_and_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, logical_and_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, logical_and_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, logical_or_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, logical_or_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, logical_or_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, logical_or_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, logical_or_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, logical_or_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, logical_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, logical_or_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, logical_or_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, logical_or_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, logical_or_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int8,bh_int8, logical_xor_functor<bh_bool,bh_int8,bh_int8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float64,bh_float64, logical_xor_functor<bh_bool,bh_float64,bh_float64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint16,bh_uint16, logical_xor_functor<bh_bool,bh_uint16,bh_uint16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint64,bh_uint64, logical_xor_functor<bh_bool,bh_uint64,bh_uint64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int16,bh_int16, logical_xor_functor<bh_bool,bh_int16,bh_int16 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_float32,bh_float32, logical_xor_functor<bh_bool,bh_float32,bh_float32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, logical_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int32,bh_int32, logical_xor_functor<bh_bool,bh_int32,bh_int32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint8,bh_uint8, logical_xor_functor<bh_bool,bh_uint8,bh_uint8 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_int64,bh_int64, logical_xor_functor<bh_bool,bh_int64,bh_int64 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_uint32,bh_uint32, logical_xor_functor<bh_bool,bh_uint32,bh_uint32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_bool,bh_int8, logical_not_functor<bh_bool,bh_int8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int8, logical_not_functor<bh_bool,bh_int8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_bool,bh_float64, logical_not_functor<bh_bool,bh_float64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float64, logical_not_functor<bh_bool,bh_float64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_bool,bh_uint16, logical_not_functor<bh_bool,bh_uint16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint16, logical_not_functor<bh_bool,bh_uint16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_bool,bh_uint64, logical_not_functor<bh_bool,bh_uint64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint64, logical_not_functor<bh_bool,bh_uint64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_bool,bh_int16, logical_not_functor<bh_bool,bh_int16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int16, logical_not_functor<bh_bool,bh_int16 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_bool,bh_float32, logical_not_functor<bh_bool,bh_float32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float32, logical_not_functor<bh_bool,bh_float32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_bool,bh_bool, logical_not_functor<bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_bool, logical_not_functor<bh_bool,bh_bool > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_bool,bh_int32, logical_not_functor<bh_bool,bh_int32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int32, logical_not_functor<bh_bool,bh_int32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_bool,bh_uint8, logical_not_functor<bh_bool,bh_uint8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint8, logical_not_functor<bh_bool,bh_uint8 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_bool,bh_int64, logical_not_functor<bh_bool,bh_int64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int64, logical_not_functor<bh_bool,bh_int64 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_bool,bh_uint32, logical_not_functor<bh_bool,bh_uint32 > >;
        case CPHVB_LOGICAL_NOT + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint32, logical_not_functor<bh_bool,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, maximum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, maximum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, maximum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, maximum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, maximum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, maximum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, maximum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, maximum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, maximum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, maximum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, maximum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, minimum_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, minimum_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, minimum_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, minimum_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, minimum_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, minimum_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, minimum_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, minimum_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, minimum_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, minimum_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, minimum_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, bitwise_and_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_and_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_and_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, bitwise_and_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, bitwise_and_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, bitwise_and_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_and_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, bitwise_and_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_and_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, bitwise_or_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_or_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_or_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, bitwise_or_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, bitwise_or_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, bitwise_or_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_or_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, bitwise_or_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_or_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, bitwise_xor_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, bitwise_xor_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, bitwise_xor_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, bitwise_xor_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16):
            return &traverse_aaa<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 18):
            return &traverse_aac<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (CPHVB_BOOL <<16) + (1 << 17):
            return &traverse_aca<bh_bool,bh_bool,bh_bool, bitwise_xor_functor<bh_bool,bh_bool,bh_bool > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, bitwise_xor_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, bitwise_xor_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, bitwise_xor_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, bitwise_xor_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_INVERT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int8,bh_int8, invert_functor<bh_int8,bh_int8 > >;
        case CPHVB_INVERT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int8, invert_functor<bh_int8,bh_int8 > >;
        case CPHVB_INVERT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint16,bh_uint16, invert_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_INVERT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint16, invert_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_INVERT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint64,bh_uint64, invert_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_INVERT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint64, invert_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_INVERT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int16,bh_int16, invert_functor<bh_int16,bh_int16 > >;
        case CPHVB_INVERT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int16, invert_functor<bh_int16,bh_int16 > >;
        case CPHVB_INVERT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_bool,bh_bool, invert_functor<bh_bool,bh_bool > >;
        case CPHVB_INVERT + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_bool, invert_functor<bh_bool,bh_bool > >;
        case CPHVB_INVERT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int32,bh_int32, invert_functor<bh_int32,bh_int32 > >;
        case CPHVB_INVERT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int32, invert_functor<bh_int32,bh_int32 > >;
        case CPHVB_INVERT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint8,bh_uint8, invert_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_INVERT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint8, invert_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_INVERT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int64,bh_int64, invert_functor<bh_int64,bh_int64 > >;
        case CPHVB_INVERT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int64, invert_functor<bh_int64,bh_int64 > >;
        case CPHVB_INVERT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint32,bh_uint32, invert_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_INVERT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint32, invert_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, left_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, left_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, left_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, left_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, left_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, left_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, left_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, left_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, right_shift_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, right_shift_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, right_shift_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, right_shift_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, right_shift_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, right_shift_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, right_shift_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, right_shift_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_COS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, cos_functor<bh_float32,bh_float32 > >;
        case CPHVB_COS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, cos_functor<bh_float32,bh_float32 > >;
        case CPHVB_COS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, cos_functor<bh_float64,bh_float64 > >;
        case CPHVB_COS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, cos_functor<bh_float64,bh_float64 > >;
        case CPHVB_COS + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, cos_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COS + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, cos_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COS + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, cos_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COS + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, cos_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, sin_functor<bh_float32,bh_float32 > >;
        case CPHVB_SIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, sin_functor<bh_float32,bh_float32 > >;
        case CPHVB_SIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, sin_functor<bh_float64,bh_float64 > >;
        case CPHVB_SIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, sin_functor<bh_float64,bh_float64 > >;
        case CPHVB_SIN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, sin_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, sin_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, sin_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SIN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, sin_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, tan_functor<bh_float32,bh_float32 > >;
        case CPHVB_TAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, tan_functor<bh_float32,bh_float32 > >;
        case CPHVB_TAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, tan_functor<bh_float64,bh_float64 > >;
        case CPHVB_TAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, tan_functor<bh_float64,bh_float64 > >;
        case CPHVB_TAN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, tan_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, tan_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, tan_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TAN + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, tan_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, cosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_COSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, cosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_COSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, cosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_COSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, cosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_COSH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, cosh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, cosh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, cosh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_COSH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, cosh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, sinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_SINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, sinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_SINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, sinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_SINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, sinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_SINH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, sinh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, sinh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, sinh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SINH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, sinh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, tanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_TANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, tanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_TANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, tanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_TANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, tanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_TANH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, tanh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, tanh_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, tanh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_TANH + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, tanh_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arcsin_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arcsin_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arcsin_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSIN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arcsin_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arccos_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arccos_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arccos_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOS + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arccos_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arctan_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arctan_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arctan_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arctan_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arcsinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arcsinh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arcsinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCSINH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arcsinh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arccosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arccosh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arccosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCCOSH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arccosh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, arctanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, arctanh_functor<bh_float32,bh_float32 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, arctanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTANH + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, arctanh_functor<bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, arctan2_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, arctan2_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, exp_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, exp_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, exp_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, exp_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, exp_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, exp_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, exp_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_EXP + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, exp_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, exp2_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, exp2_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, exp2_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXP2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, exp2_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, expm1_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, expm1_functor<bh_float32,bh_float32 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, expm1_functor<bh_float64,bh_float64 > >;
        case CPHVB_EXPM1 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, expm1_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, log_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, log_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, log_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, log_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, log_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, log_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, log_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, log_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, log2_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, log2_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, log2_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG2 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, log2_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, log10_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, log10_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, log10_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, log10_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, log10_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, log10_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, log10_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG10 + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, log10_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, log1p_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, log1p_functor<bh_float32,bh_float32 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, log1p_functor<bh_float64,bh_float64 > >;
        case CPHVB_LOG1P + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, log1p_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, sqrt_functor<bh_float32,bh_float32 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, sqrt_functor<bh_float32,bh_float32 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, sqrt_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, sqrt_functor<bh_float64,bh_float64 > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, sqrt_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, sqrt_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, sqrt_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_SQRT + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, sqrt_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_CEIL + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, ceil_functor<bh_float32,bh_float32 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, ceil_functor<bh_float32,bh_float32 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, ceil_functor<bh_float64,bh_float64 > >;
        case CPHVB_CEIL + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, ceil_functor<bh_float64,bh_float64 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, trunc_functor<bh_float32,bh_float32 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, trunc_functor<bh_float32,bh_float32 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, trunc_functor<bh_float64,bh_float64 > >;
        case CPHVB_TRUNC + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, trunc_functor<bh_float64,bh_float64 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, floor_functor<bh_float32,bh_float32 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, floor_functor<bh_float32,bh_float32 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, floor_functor<bh_float64,bh_float64 > >;
        case CPHVB_FLOOR + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, floor_functor<bh_float64,bh_float64 > >;
        case CPHVB_RINT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, rint_functor<bh_float32,bh_float32 > >;
        case CPHVB_RINT + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, rint_functor<bh_float32,bh_float32 > >;
        case CPHVB_RINT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, rint_functor<bh_float64,bh_float64 > >;
        case CPHVB_RINT + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, rint_functor<bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16):
            return &traverse_aaa<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 18):
            return &traverse_aac<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (CPHVB_INT8 <<16) + (1 << 17):
            return &traverse_aca<bh_int8,bh_int8,bh_int8, mod_functor<bh_int8,bh_int8,bh_int8 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16):
            return &traverse_aaa<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 18):
            return &traverse_aac<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (CPHVB_FLOAT64 <<16) + (1 << 17):
            return &traverse_aca<bh_float64,bh_float64,bh_float64, mod_functor<bh_float64,bh_float64,bh_float64 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16):
            return &traverse_aaa<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 18):
            return &traverse_aac<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (CPHVB_UINT16 <<16) + (1 << 17):
            return &traverse_aca<bh_uint16,bh_uint16,bh_uint16, mod_functor<bh_uint16,bh_uint16,bh_uint16 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16):
            return &traverse_aaa<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 18):
            return &traverse_aac<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (CPHVB_UINT64 <<16) + (1 << 17):
            return &traverse_aca<bh_uint64,bh_uint64,bh_uint64, mod_functor<bh_uint64,bh_uint64,bh_uint64 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16):
            return &traverse_aaa<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 18):
            return &traverse_aac<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (CPHVB_INT16 <<16) + (1 << 17):
            return &traverse_aca<bh_int16,bh_int16,bh_int16, mod_functor<bh_int16,bh_int16,bh_int16 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16):
            return &traverse_aaa<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 18):
            return &traverse_aac<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (CPHVB_FLOAT32 <<16) + (1 << 17):
            return &traverse_aca<bh_float32,bh_float32,bh_float32, mod_functor<bh_float32,bh_float32,bh_float32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16):
            return &traverse_aaa<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 18):
            return &traverse_aac<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (CPHVB_INT32 <<16) + (1 << 17):
            return &traverse_aca<bh_int32,bh_int32,bh_int32, mod_functor<bh_int32,bh_int32,bh_int32 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16):
            return &traverse_aaa<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 18):
            return &traverse_aac<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (CPHVB_UINT8 <<16) + (1 << 17):
            return &traverse_aca<bh_uint8,bh_uint8,bh_uint8, mod_functor<bh_uint8,bh_uint8,bh_uint8 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16):
            return &traverse_aaa<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 18):
            return &traverse_aac<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (CPHVB_INT64 <<16) + (1 << 17):
            return &traverse_aca<bh_int64,bh_int64,bh_int64, mod_functor<bh_int64,bh_int64,bh_int64 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16):
            return &traverse_aaa<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 18):
            return &traverse_aac<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_MOD + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (CPHVB_UINT32 <<16) + (1 << 17):
            return &traverse_aca<bh_uint32,bh_uint32,bh_uint32, mod_functor<bh_uint32,bh_uint32,bh_uint32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_bool,bh_float32, isnan_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float32, isnan_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_bool,bh_float64, isnan_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISNAN + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float64, isnan_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_bool,bh_float32, isinf_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float32, isinf_functor<bh_bool,bh_float32 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_bool,bh_float64, isinf_functor<bh_bool,bh_float64 > >;
        case CPHVB_ISINF + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float64, isinf_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_bool,bh_int8, identity_functor<bh_bool,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int8, identity_functor<bh_bool,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int8,bh_int8, identity_functor<bh_int8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int8, identity_functor<bh_int8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_uint8,bh_int8, identity_functor<bh_uint8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_int8, identity_functor<bh_uint8,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int16,bh_int8, identity_functor<bh_int16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int8, identity_functor<bh_int16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_uint16,bh_int8, identity_functor<bh_uint16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_int8, identity_functor<bh_uint16,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int32,bh_int8, identity_functor<bh_int32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int8, identity_functor<bh_int32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_uint32,bh_int8, identity_functor<bh_uint32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_int8, identity_functor<bh_uint32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_int64,bh_int8, identity_functor<bh_int64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int8, identity_functor<bh_int64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_uint64,bh_int8, identity_functor<bh_uint64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_int8, identity_functor<bh_uint64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_float32,bh_int8, identity_functor<bh_float32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_int8, identity_functor<bh_float32,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<bh_float64,bh_int8, identity_functor<bh_float64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_int8, identity_functor<bh_float64,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_bool,bh_float64, identity_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float64, identity_functor<bh_bool,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_int8,bh_float64, identity_functor<bh_int8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_float64, identity_functor<bh_int8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_uint8,bh_float64, identity_functor<bh_uint8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_float64, identity_functor<bh_uint8,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_int16,bh_float64, identity_functor<bh_int16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_float64, identity_functor<bh_int16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_uint16,bh_float64, identity_functor<bh_uint16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_float64, identity_functor<bh_uint16,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_int32,bh_float64, identity_functor<bh_int32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_float64, identity_functor<bh_int32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_uint32,bh_float64, identity_functor<bh_uint32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_float64, identity_functor<bh_uint32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_int64,bh_float64, identity_functor<bh_int64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_float64, identity_functor<bh_int64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_uint64,bh_float64, identity_functor<bh_uint64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_float64, identity_functor<bh_uint64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float32,bh_float64, identity_functor<bh_float32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float64, identity_functor<bh_float32,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<bh_float64,bh_float64, identity_functor<bh_float64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float64, identity_functor<bh_float64,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_bool,bh_uint16, identity_functor<bh_bool,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint16, identity_functor<bh_bool,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_int8,bh_uint16, identity_functor<bh_int8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_uint16, identity_functor<bh_int8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint8,bh_uint16, identity_functor<bh_uint8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint16, identity_functor<bh_uint8,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_int16,bh_uint16, identity_functor<bh_int16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_uint16, identity_functor<bh_int16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint16,bh_uint16, identity_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint16, identity_functor<bh_uint16,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_int32,bh_uint16, identity_functor<bh_int32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_uint16, identity_functor<bh_int32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint32,bh_uint16, identity_functor<bh_uint32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint16, identity_functor<bh_uint32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_int64,bh_uint16, identity_functor<bh_int64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_uint16, identity_functor<bh_int64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_uint64,bh_uint16, identity_functor<bh_uint64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint16, identity_functor<bh_uint64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_float32,bh_uint16, identity_functor<bh_float32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_uint16, identity_functor<bh_float32,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<bh_float64,bh_uint16, identity_functor<bh_float64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_uint16, identity_functor<bh_float64,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_bool,bh_uint64, identity_functor<bh_bool,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint64, identity_functor<bh_bool,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_int8,bh_uint64, identity_functor<bh_int8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_uint64, identity_functor<bh_int8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint8,bh_uint64, identity_functor<bh_uint8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint64, identity_functor<bh_uint8,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_int16,bh_uint64, identity_functor<bh_int16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_uint64, identity_functor<bh_int16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint16,bh_uint64, identity_functor<bh_uint16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint64, identity_functor<bh_uint16,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_int32,bh_uint64, identity_functor<bh_int32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_uint64, identity_functor<bh_int32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint32,bh_uint64, identity_functor<bh_uint32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint64, identity_functor<bh_uint32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_int64,bh_uint64, identity_functor<bh_int64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_uint64, identity_functor<bh_int64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_uint64,bh_uint64, identity_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint64, identity_functor<bh_uint64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_float32,bh_uint64, identity_functor<bh_float32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_uint64, identity_functor<bh_float32,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<bh_float64,bh_uint64, identity_functor<bh_float64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_uint64, identity_functor<bh_float64,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_bool,bh_int16, identity_functor<bh_bool,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int16, identity_functor<bh_bool,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int8,bh_int16, identity_functor<bh_int8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int16, identity_functor<bh_int8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_uint8,bh_int16, identity_functor<bh_uint8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_int16, identity_functor<bh_uint8,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int16,bh_int16, identity_functor<bh_int16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int16, identity_functor<bh_int16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_uint16,bh_int16, identity_functor<bh_uint16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_int16, identity_functor<bh_uint16,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int32,bh_int16, identity_functor<bh_int32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int16, identity_functor<bh_int32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_uint32,bh_int16, identity_functor<bh_uint32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_int16, identity_functor<bh_uint32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_int64,bh_int16, identity_functor<bh_int64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int16, identity_functor<bh_int64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_uint64,bh_int16, identity_functor<bh_uint64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_int16, identity_functor<bh_uint64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_float32,bh_int16, identity_functor<bh_float32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_int16, identity_functor<bh_float32,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<bh_float64,bh_int16, identity_functor<bh_float64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_int16, identity_functor<bh_float64,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_bool,bh_float32, identity_functor<bh_bool,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_float32, identity_functor<bh_bool,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_int8,bh_float32, identity_functor<bh_int8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_float32, identity_functor<bh_int8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_uint8,bh_float32, identity_functor<bh_uint8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_float32, identity_functor<bh_uint8,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_int16,bh_float32, identity_functor<bh_int16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_float32, identity_functor<bh_int16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_uint16,bh_float32, identity_functor<bh_uint16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_float32, identity_functor<bh_uint16,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_int32,bh_float32, identity_functor<bh_int32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_float32, identity_functor<bh_int32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_uint32,bh_float32, identity_functor<bh_uint32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_float32, identity_functor<bh_uint32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_int64,bh_float32, identity_functor<bh_int64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_float32, identity_functor<bh_int64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_uint64,bh_float32, identity_functor<bh_uint64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_float32, identity_functor<bh_uint64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float32,bh_float32, identity_functor<bh_float32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_float32, identity_functor<bh_float32,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<bh_float64,bh_float32, identity_functor<bh_float64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_float32, identity_functor<bh_float64,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_bool,bh_bool, identity_functor<bh_bool,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_bool, identity_functor<bh_bool,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_int8,bh_bool, identity_functor<bh_int8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_bool, identity_functor<bh_int8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_uint8,bh_bool, identity_functor<bh_uint8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_bool, identity_functor<bh_uint8,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_int16,bh_bool, identity_functor<bh_int16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_bool, identity_functor<bh_int16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_uint16,bh_bool, identity_functor<bh_uint16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_bool, identity_functor<bh_uint16,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_int32,bh_bool, identity_functor<bh_int32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_bool, identity_functor<bh_int32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_uint32,bh_bool, identity_functor<bh_uint32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_bool, identity_functor<bh_uint32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_int64,bh_bool, identity_functor<bh_int64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_bool, identity_functor<bh_int64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_uint64,bh_bool, identity_functor<bh_uint64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_bool, identity_functor<bh_uint64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_float32,bh_bool, identity_functor<bh_float32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_bool, identity_functor<bh_float32,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<bh_float64,bh_bool, identity_functor<bh_float64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_bool, identity_functor<bh_float64,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_bool,bh_int32, identity_functor<bh_bool,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int32, identity_functor<bh_bool,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int8,bh_int32, identity_functor<bh_int8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int32, identity_functor<bh_int8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_uint8,bh_int32, identity_functor<bh_uint8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_int32, identity_functor<bh_uint8,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int16,bh_int32, identity_functor<bh_int16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int32, identity_functor<bh_int16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_uint16,bh_int32, identity_functor<bh_uint16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_int32, identity_functor<bh_uint16,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int32,bh_int32, identity_functor<bh_int32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int32, identity_functor<bh_int32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_uint32,bh_int32, identity_functor<bh_uint32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_int32, identity_functor<bh_uint32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_int64,bh_int32, identity_functor<bh_int64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int32, identity_functor<bh_int64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_uint64,bh_int32, identity_functor<bh_uint64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_int32, identity_functor<bh_uint64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_float32,bh_int32, identity_functor<bh_float32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_int32, identity_functor<bh_float32,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<bh_float64,bh_int32, identity_functor<bh_float64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_int32, identity_functor<bh_float64,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_bool,bh_uint8, identity_functor<bh_bool,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint8, identity_functor<bh_bool,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_int8,bh_uint8, identity_functor<bh_int8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_uint8, identity_functor<bh_int8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint8,bh_uint8, identity_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint8, identity_functor<bh_uint8,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_int16,bh_uint8, identity_functor<bh_int16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_uint8, identity_functor<bh_int16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint16,bh_uint8, identity_functor<bh_uint16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint8, identity_functor<bh_uint16,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_int32,bh_uint8, identity_functor<bh_int32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_uint8, identity_functor<bh_int32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint32,bh_uint8, identity_functor<bh_uint32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint8, identity_functor<bh_uint32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_int64,bh_uint8, identity_functor<bh_int64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_uint8, identity_functor<bh_int64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_uint64,bh_uint8, identity_functor<bh_uint64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint8, identity_functor<bh_uint64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_float32,bh_uint8, identity_functor<bh_float32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_uint8, identity_functor<bh_float32,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<bh_float64,bh_uint8, identity_functor<bh_float64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_uint8, identity_functor<bh_float64,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_bool,bh_int64, identity_functor<bh_bool,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_int64, identity_functor<bh_bool,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int8,bh_int64, identity_functor<bh_int8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_int64, identity_functor<bh_int8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_uint8,bh_int64, identity_functor<bh_uint8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_int64, identity_functor<bh_uint8,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int16,bh_int64, identity_functor<bh_int16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_int64, identity_functor<bh_int16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_uint16,bh_int64, identity_functor<bh_uint16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_int64, identity_functor<bh_uint16,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int32,bh_int64, identity_functor<bh_int32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_int64, identity_functor<bh_int32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_uint32,bh_int64, identity_functor<bh_uint32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_int64, identity_functor<bh_uint32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_int64,bh_int64, identity_functor<bh_int64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_int64, identity_functor<bh_int64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_uint64,bh_int64, identity_functor<bh_uint64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_int64, identity_functor<bh_uint64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_float32,bh_int64, identity_functor<bh_float32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_int64, identity_functor<bh_float32,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<bh_float64,bh_int64, identity_functor<bh_float64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_int64, identity_functor<bh_float64,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_bool,bh_uint32, identity_functor<bh_bool,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_BOOL << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_bool,bh_uint32, identity_functor<bh_bool,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_int8,bh_uint32, identity_functor<bh_int8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT8 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int8,bh_uint32, identity_functor<bh_int8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint8,bh_uint32, identity_functor<bh_uint8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT8 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint8,bh_uint32, identity_functor<bh_uint8,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_int16,bh_uint32, identity_functor<bh_int16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT16 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int16,bh_uint32, identity_functor<bh_int16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint16,bh_uint32, identity_functor<bh_uint16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT16 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint16,bh_uint32, identity_functor<bh_uint16,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_int32,bh_uint32, identity_functor<bh_int32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int32,bh_uint32, identity_functor<bh_int32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint32,bh_uint32, identity_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint32,bh_uint32, identity_functor<bh_uint32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_int64,bh_uint32, identity_functor<bh_int64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_INT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_int64,bh_uint32, identity_functor<bh_int64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_uint64,bh_uint32, identity_functor<bh_uint64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_UINT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_uint64,bh_uint32, identity_functor<bh_uint64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_float32,bh_uint32, identity_functor<bh_float32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT32 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float32,bh_uint32, identity_functor<bh_float32,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<bh_float64,bh_uint32, identity_functor<bh_float64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_FLOAT64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<bh_float64,bh_uint32, identity_functor<bh_float64,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<float>,std::complex<float>, identity_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<float>, identity_functor<std::complex<float>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<double>,std::complex<double>, identity_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<double>, identity_functor<std::complex<double>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<std::complex<float>,bh_bool, identity_functor<std::complex<float>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_bool, identity_functor<std::complex<float>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_BOOL <<12):
            return &traverse_aa<std::complex<double>,bh_bool, identity_functor<std::complex<double>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_BOOL <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_bool, identity_functor<std::complex<double>,bh_bool > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<std::complex<float>,bh_int8, identity_functor<std::complex<float>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_int8, identity_functor<std::complex<float>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT8 <<12):
            return &traverse_aa<std::complex<double>,bh_int8, identity_functor<std::complex<double>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT8 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_int8, identity_functor<std::complex<double>,bh_int8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<std::complex<float>,bh_uint8, identity_functor<std::complex<float>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_uint8, identity_functor<std::complex<float>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT8 <<12):
            return &traverse_aa<std::complex<double>,bh_uint8, identity_functor<std::complex<double>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT8 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_uint8, identity_functor<std::complex<double>,bh_uint8 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<std::complex<float>,bh_int16, identity_functor<std::complex<float>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_int16, identity_functor<std::complex<float>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT16 <<12):
            return &traverse_aa<std::complex<double>,bh_int16, identity_functor<std::complex<double>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT16 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_int16, identity_functor<std::complex<double>,bh_int16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<std::complex<float>,bh_uint16, identity_functor<std::complex<float>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_uint16, identity_functor<std::complex<float>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT16 <<12):
            return &traverse_aa<std::complex<double>,bh_uint16, identity_functor<std::complex<double>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT16 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_uint16, identity_functor<std::complex<double>,bh_uint16 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<std::complex<float>,bh_int32, identity_functor<std::complex<float>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_int32, identity_functor<std::complex<float>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT32 <<12):
            return &traverse_aa<std::complex<double>,bh_int32, identity_functor<std::complex<double>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_int32, identity_functor<std::complex<double>,bh_int32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<std::complex<float>,bh_uint32, identity_functor<std::complex<float>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_uint32, identity_functor<std::complex<float>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT32 <<12):
            return &traverse_aa<std::complex<double>,bh_uint32, identity_functor<std::complex<double>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_uint32, identity_functor<std::complex<double>,bh_uint32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<std::complex<float>,bh_int64, identity_functor<std::complex<float>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_int64, identity_functor<std::complex<float>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT64 <<12):
            return &traverse_aa<std::complex<double>,bh_int64, identity_functor<std::complex<double>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_INT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_int64, identity_functor<std::complex<double>,bh_int64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<std::complex<float>,bh_uint64, identity_functor<std::complex<float>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_uint64, identity_functor<std::complex<float>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT64 <<12):
            return &traverse_aa<std::complex<double>,bh_uint64, identity_functor<std::complex<double>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_UINT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_uint64, identity_functor<std::complex<double>,bh_uint64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<std::complex<float>,bh_float32, identity_functor<std::complex<float>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_float32, identity_functor<std::complex<float>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT32 <<12):
            return &traverse_aa<std::complex<double>,bh_float32, identity_functor<std::complex<double>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT32 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_float32, identity_functor<std::complex<double>,bh_float32 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<std::complex<float>,bh_float64, identity_functor<std::complex<float>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,bh_float64, identity_functor<std::complex<float>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT64 <<12):
            return &traverse_aa<std::complex<double>,bh_float64, identity_functor<std::complex<double>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_FLOAT64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,bh_float64, identity_functor<std::complex<double>,bh_float64 > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX64 <<12):
            return &traverse_aa<std::complex<double>,std::complex<float>, identity_functor<std::complex<double>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX128 << 8) +(CPHVB_COMPLEX64 <<12) + (1 << 17):
            return &traverse_ac<std::complex<double>,std::complex<float>, identity_functor<std::complex<double>,std::complex<float> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX128 <<12):
            return &traverse_aa<std::complex<float>,std::complex<double>, identity_functor<std::complex<float>,std::complex<double> > >;
        case CPHVB_IDENTITY + (CPHVB_COMPLEX64 << 8) +(CPHVB_COMPLEX128 <<12) + (1 << 17):
            return &traverse_ac<std::complex<float>,std::complex<double>, identity_functor<std::complex<float>,std::complex<double> > >;

        default:
            return NULL;

    }

}


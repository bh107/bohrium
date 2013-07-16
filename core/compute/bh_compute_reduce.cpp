


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
#include <bh_vcache.h>
#include "functors.hpp"
#include <complex>
#include <assert.h>
#include "functors.hpp"
#include "traverser.hpp"

typedef char BYTE;

// These are legacy

template <typename T, typename Instr>
bh_error bh_compute_reduce_any_naive( bh_view* op_out, bh_view* op_in, bh_index axis, bh_opcode opcode )
{
    Instr opcode_func;                          // Functor-pointer
                                                // Data pointers
    T* data_out = (T*) bh_base_array(op_out)->data;
    T* data_in  = (T*) bh_base_array(op_in)->data;

    bh_index stride      = op_in->stride[axis];
    bh_index nelements   = op_in->shape[axis];
    bh_index i, j, off0;

    if (op_in->ndim == 1) {                     // 1D special case

        data_out += op_out->start;
        *data_out = *(data_in+op_in->start);    // Initialize pseudo-scalar output
                                                // the value of the first element
                                                // in input.
        for(off0 = op_in->start+stride, j=1; j < nelements; j++, off0 += stride ) {
            opcode_func( data_out, data_out, (data_in+off0) );
        }

        return BH_SUCCESS;

    } else {                                    // ND general case

        bh_instruction instr;
        bh_error err;

        bh_view *tmp = &instr.operand[1];  // Copy the input-array meta-data
        tmp->base    = bh_base_array(op_in);
        tmp->ndim    = op_in->ndim-1;
        tmp->start   = op_in->start;

        for(j=0, i=0; i < op_in->ndim; ++i) {        // Copy every dimension except for the 'axis' dimension.
            if(i != axis) {
                tmp->shape[j]    = op_in->shape[i];
                tmp->stride[j]   = op_in->stride[i];
                ++j;
            }
        }

        instr.opcode = BH_IDENTITY;       // Copy the first element to the output.
        instr.operand[0] = *op_out;

        err = bh_compute_apply_naive( &instr );
        if (err != BH_SUCCESS) {
            return err;
        }
        tmp->start += stride;

        instr.opcode = opcode;                   // Reduce over the 'axis' dimension.
        instr.operand[0] = *op_out;              // NB: the first element is already handled.
        instr.operand[2] = *op_out;              // Note that operand[1] is still the tmp view

        for(i=1; i<nelements; ++i) {
            err = bh_compute_apply_naive( &instr );
            if (err != BH_SUCCESS) {
                return err;
            }
            tmp->start += stride;
        }
        return BH_SUCCESS;
    }
}

bh_error bh_compute_reduce_naive(bh_instruction *inst)
{
    if (inst->constant.type != BH_INT64)
        return BH_TYPE_NOT_SUPPORTED;

    bh_index axis = inst->constant.value.int64;
    bh_opcode opcode;
    bh_view *op_out = &inst->operand[0];
    bh_view *op_in  = &inst->operand[1];

    assert(op_out != NULL);

    switch (inst->opcode)
    {
        case BH_ADD_REDUCE:
            opcode = BH_ADD;
            break;
        case BH_MULTIPLY_REDUCE:
            opcode = BH_MULTIPLY;
            break;
        case BH_MINIMUM_REDUCE:
            opcode = BH_MINIMUM;
            break;
        case BH_MAXIMUM_REDUCE:
            opcode = BH_MAXIMUM;
            break;
        case BH_LOGICAL_AND_REDUCE:
            opcode = BH_LOGICAL_AND;
            break;
        case BH_BITWISE_AND_REDUCE:
            opcode = BH_BITWISE_AND;
            break;
        case BH_LOGICAL_OR_REDUCE:
            opcode = BH_LOGICAL_OR;
            break;
        case BH_BITWISE_OR_REDUCE:
            opcode = BH_BITWISE_OR;
            break;
        case BH_LOGICAL_XOR_REDUCE:
            opcode = BH_LOGICAL_XOR;
            break;
        case BH_BITWISE_XOR_REDUCE:
            opcode = BH_BITWISE_XOR;
            break;
        default:
            return BH_TYPE_NOT_SUPPORTED;
    }

                                                        //  Sanity checks.
    if (bh_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %lld is not a binary ufunc, hence it is not suitable for reduction.\n", (long long int)opcode);
        return BH_ERROR;
    }

    if (bh_base_array(op_in)->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op_in ) is null.\n");
        return BH_ERROR;
    }

    if (op_in->base->type != op_out->base->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return BH_ERROR;
    }

    long int poly = opcode + (op_in->base->type << 8);

    switch(poly) {

        case BH_ADD + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, add_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, add_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, add_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, add_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, add_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, add_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, add_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, add_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, add_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, add_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, add_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, add_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, add_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, multiply_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, multiply_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, multiply_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, multiply_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, multiply_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, multiply_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, multiply_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, multiply_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, multiply_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, multiply_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, multiply_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, multiply_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, multiply_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, minimum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, minimum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, minimum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, minimum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, minimum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, minimum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, minimum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, minimum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, minimum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, minimum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, minimum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, maximum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, maximum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, maximum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, maximum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, maximum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, maximum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, maximum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, maximum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, maximum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, maximum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, maximum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:

            return BH_ERROR;

    }

}

// These are the future

template <typename T, typename Instr>
bh_error bh_compute_reduce_any( bh_view* op_out, bh_view* op_in, bh_index axis, bh_opcode opcode )
{
    Instr opcode_func;                          // Functor-pointer
                                                // Data pointers
    T* data_out = (T*) bh_base_array(op_out)->data;
    T* data_in  = (T*) bh_base_array(op_in)->data;

    bh_index nelements   = op_in->shape[axis];
    bh_index i, j;

    bh_index el_size = sizeof(T);

    if (op_in->ndim == 1) {                     // 1D special case

        data_out += op_out->start;
        *data_out = *(data_in+op_in->start);    // Initialize pseudo-scalar output
                                                // the value of the first element
                                                // in input.
        bh_index stride = op_in->stride[axis] * el_size;
        BYTE* src = ((BYTE*)(data_in + op_in->start)) + stride;

        // 4x Loop unrolling
        bh_index fulls = (nelements - 1) / 4;
        bh_index remainder = (nelements - 1) % 4;

        for (i = 0; i < fulls; i++) {
            opcode_func( data_out, data_out, ((T*)src) );
            src += stride;
            opcode_func( data_out, data_out, ((T*)src) );
            src += stride;
            opcode_func( data_out, data_out, ((T*)src) );
            src += stride;
            opcode_func( data_out, data_out, ((T*)src) );
            src += stride;
        }

        switch (remainder) {
            case 3:
                opcode_func( data_out, data_out, ((T*)src) );
                src += stride;
            case 2:
                opcode_func( data_out, data_out, ((T*)src) );
                src += stride;
            case 1:
                opcode_func( data_out, data_out, ((T*)src) );
                src += stride;
        }

    /*
    } else if (op_in->ndim == 2) {          // 2D General case

        Experiments show that the general case is almost as fast as
         an optimized 2D version
    */

    } else {                                    // ND general case

        bh_instruction instr;
        bh_error err;

        bh_view *tmp = &instr.operand[1];           // Copy the input-array meta-data
        tmp->base    = bh_base_array(op_in);
        tmp->ndim    = op_in->ndim-1;
        tmp->start   = op_in->start;
        for(j=0, i=0; i < op_in->ndim; ++i) {        // Copy every dimension except for the 'axis' dimension.
            if(i != axis) {
                tmp->shape[j]    = op_in->shape[i];
                tmp->stride[j]   = op_in->stride[i];
                ++j;
            }
        }

        instr.opcode = BH_IDENTITY;                 // Copy the first element to the output.
        instr.operand[0] = *op_out;
        bh_tstate state;
        bh_tstate_reset( &state, &instr );
        err = traverse_aa<T, T, identity_functor<T, T> >(&instr, &state);
        if (err != BH_SUCCESS) {
            return err;
        }

        bh_index stride = op_in->stride[axis];
        bh_index stride_bytes = stride * sizeof(T);

        instr.opcode = opcode;                // Reduce over the 'axis' dimension.
        instr.operand[0] = *op_out;           // NB: the first element is already handled.
        instr.operand[2] = *op_out;           // Note that operand[1] is still the tmp view

        tmp->start += stride;
        bh_tstate_reset( &state, &instr );

        void* out_start = state.start[0];
        void* tmp_start = state.start[1];

        for(i=1; i<nelements; ++i) {

            err = traverse_aaa<T, T, T, Instr>(&instr, &state);
            if (err != BH_SUCCESS) {
                return err;
            }

            tmp->start += stride;

            // Faster replacement of bh_tstate_reset
            tmp_start = (void*)(((char*)tmp_start) + stride_bytes);
            state.start[0] = out_start;
            state.start[1] = tmp_start;
            state.start[2] = out_start;
        }
    }
    return BH_SUCCESS;
}

bh_error bh_compute_reduce(bh_instruction *inst)
{
    if (inst->constant.type != BH_INT64)
        return BH_TYPE_NOT_SUPPORTED;

    bh_index axis = inst->constant.value.int64;
    bh_opcode opcode;
    bh_view *op_out = &inst->operand[0];
    bh_view *op_in  = &inst->operand[1];

    assert(op_out != NULL);

    switch (inst->opcode)
    {
        case BH_ADD_REDUCE:
            opcode = BH_ADD;
            break;
        case BH_MULTIPLY_REDUCE:
            opcode = BH_MULTIPLY;
            break;
        case BH_MINIMUM_REDUCE:
            opcode = BH_MINIMUM;
            break;
        case BH_MAXIMUM_REDUCE:
            opcode = BH_MAXIMUM;
            break;
        case BH_LOGICAL_AND_REDUCE:
            opcode = BH_LOGICAL_AND;
            break;
        case BH_BITWISE_AND_REDUCE:
            opcode = BH_BITWISE_AND;
            break;
        case BH_LOGICAL_OR_REDUCE:
            opcode = BH_LOGICAL_OR;
            break;
        case BH_BITWISE_OR_REDUCE:
            opcode = BH_BITWISE_OR;
            break;
        case BH_LOGICAL_XOR_REDUCE:
            opcode = BH_LOGICAL_XOR;
            break;
        case BH_BITWISE_XOR_REDUCE:
            opcode = BH_BITWISE_XOR;
            break;
        default:
            return BH_TYPE_NOT_SUPPORTED;
    }

                                                        //  Sanity checks.
    if (bh_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %lld is not a binary ufunc, hence it is not suitable for reduction.\n", (long long int)opcode);
        return BH_ERROR;
    }

    if (bh_base_array(op_in)->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op_in ) is null.\n");
        return BH_ERROR;
    }

    if (op_in->base->type != op_out->base->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return BH_ERROR;
    }

    long int poly = opcode + (op_in->base->type << 8);

    switch(poly) {

        case BH_ADD + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, add_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, add_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, add_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, add_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, add_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, add_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, add_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, add_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, add_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, add_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, add_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, add_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_ADD + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, add_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, multiply_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, multiply_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, multiply_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, multiply_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, multiply_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, multiply_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, multiply_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, multiply_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, multiply_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, multiply_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, multiply_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, multiply_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MULTIPLY + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, multiply_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, minimum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, minimum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, minimum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, minimum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, minimum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, minimum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, minimum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, minimum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, minimum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, minimum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MINIMUM + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, minimum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, maximum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, maximum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, maximum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, maximum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, maximum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, maximum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, maximum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, maximum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, maximum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, maximum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MAXIMUM + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, maximum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_AND + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_OR + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_BITWISE_XOR + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:

            return BH_ERROR;

    }

}

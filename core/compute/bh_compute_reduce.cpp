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
#include <complex>
#include "functors.hpp"
#include "traverser.hpp"

typedef char BYTE;

// These are legacy

template <typename T, typename Instr>
bh_error bh_compute_reduce_any_naive( bh_array* op_out, bh_array* op_in, bh_index axis, bh_opcode opcode )
{
    Instr opcode_func;                          // Functor-pointer
                                                // Data pointers
    T* data_out = (T*) bh_base_array(op_out)->data;
    T* data_in  = (T*) bh_base_array(op_in)->data;

    bh_index stride      = op_in->stride[axis];
    bh_index nelements   = op_in->shape[axis];
    bh_index i, j, off0;

    if (op_in->ndim == 1) {                     // 1D special case

        *data_out = *(data_in+op_in->start);    // Initialize pseudo-scalar output
                                                // the value of the first element
                                                // in input.
        for(off0 = op_in->start+stride, j=1; j < nelements; j++, off0 += stride ) {
            opcode_func( data_out, data_out, (data_in+off0) );
        }

        return BH_SUCCESS;

    } else {                                    // ND general case

        bh_array tmp;                        // Copy the input-array meta-data
        bh_instruction instr; 
        bh_error err;

        tmp.base    = bh_base_array(op_in);
        tmp.type    = op_in->type;
        tmp.ndim    = op_in->ndim-1;
        tmp.start   = op_in->start;

        for(j=0, i=0; i < op_in->ndim; ++i) {        // Copy every dimension except for the 'axis' dimension.
            if(i != axis) {
                tmp.shape[j]    = op_in->shape[i];
                tmp.stride[j]   = op_in->stride[i];
                ++j;
            }
        }
        tmp.data = op_in->data;
       
        instr.status = BH_INST_PENDING;       // Copy the first element to the output.
        instr.opcode = BH_IDENTITY;
        instr.operand[0] = op_out;
        instr.operand[1] = &tmp;
        instr.operand[2] = NULL;

        // TODO: use traverse directly
        //err = traverse_aa<T, T, Instr>( instr );// execute the pseudo-instruction
        //err = bh_compute_apply( &instr );
        err = bh_compute_apply_naive( &instr );
        if (err != BH_SUCCESS) {
            return err;
        }
        tmp.start += stride;

        instr.status = BH_INST_PENDING;       // Reduce over the 'axis' dimension.
        instr.opcode = opcode;                   // NB: the first element is already handled.
        instr.operand[0] = op_out;
        instr.operand[1] = op_out;
        instr.operand[2] = &tmp;
        
        for(i=1; i<nelements; ++i) {
            // TODO: use traverse directly
            //err = traverse_aaa<T, T, T, Instr>( instr );
            err = bh_compute_apply_naive( &instr );
            if (err != BH_SUCCESS) {
                return err;
            }
            tmp.start += stride;
        }

        return BH_SUCCESS;
    }

}

bh_error bh_compute_reduce_naive(bh_userfunc *arg, void* ve_arg)
{
    bh_reduce_type *a = (bh_reduce_type *) arg;   // Grab function arguments

    bh_opcode opcode = a->opcode;                    // Opcode
    bh_index axis    = a->axis;                      // The axis to reduce "around"

    bh_array *op_out = a->operand[0];                // Output operand
    bh_array *op_in  = a->operand[1];                // Input operand

                                                        //  Sanity checks.
    if (bh_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %lld is not a binary ufunc, hence it is not suitable for reduction.\n", (long long int)opcode);
        return BH_ERROR;
    }

	if (bh_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op[1] ) is null.\n");
        return BH_ERROR;
	}

    if (op_in->type != op_out->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return BH_ERROR;
    }
    
    if (bh_data_malloc(op_out) != BH_SUCCESS) {
        fprintf(stderr, "ERR: bh_compute_reduce; No memory for reduction-result.\n");
        return BH_OUT_OF_MEMORY;
    }

    long int poly = opcode + (op_in->type << 8);

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
        case BH_SUBTRACT + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, subtract_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, subtract_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, subtract_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, subtract_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, subtract_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, subtract_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, subtract_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, subtract_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, subtract_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, subtract_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, subtract_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, subtract_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, subtract_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
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
        case BH_DIVIDE + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, divide_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, divide_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, divide_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, divide_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, divide_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, divide_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, divide_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, divide_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, divide_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, divide_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, divide_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, divide_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, power_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, power_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, power_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, power_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, power_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, power_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, power_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, power_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, power_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, power_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, greater_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, greater_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, greater_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, greater_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, greater_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, greater_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, greater_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, greater_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, greater_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, greater_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, greater_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, greater_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, greater_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, greater_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, greater_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, greater_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, greater_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, greater_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, greater_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, greater_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, greater_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, greater_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, less_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, less_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, less_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, less_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, less_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, less_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, less_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, less_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, less_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, less_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, less_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, less_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, less_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, less_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, less_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, less_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, less_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, less_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, less_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, less_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, less_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, less_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, not_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, not_equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, not_equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, not_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, not_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, not_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, not_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, not_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, not_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, not_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, not_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, not_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, not_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_and_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_and_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_or_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_or_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_xor_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_xor_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
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
        case BH_LEFT_SHIFT + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, left_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, left_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, left_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, left_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, left_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, left_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, left_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, left_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, right_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, right_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, right_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, right_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, right_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, right_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, right_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, right_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_ARCTAN2 + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, arctan2_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_ARCTAN2 + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, arctan2_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, mod_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, mod_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, mod_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, mod_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, mod_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, mod_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, mod_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, mod_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, mod_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, mod_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:
            
            return BH_ERROR;

    }

}

// These are the future

template <typename T, typename Instr>
bh_error bh_compute_reduce_any( bh_array* op_out, bh_array* op_in, bh_index axis, bh_opcode opcode )
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
	} else if (op_in->ndim == 2) {			// 2D General case

		Experiments show that the general case is almost as fast as
		 an optimized 2D version
	*/
    
    } else {                                    // ND general case

        bh_array tmp;                        // Copy the input-array meta-data
        bh_instruction instr; 
        bh_error err;

        tmp.base    = bh_base_array(op_in);
        tmp.type    = op_in->type;
        tmp.ndim    = op_in->ndim-1;
        tmp.start   = op_in->start;

        for(j=0, i=0; i < op_in->ndim; ++i) {        // Copy every dimension except for the 'axis' dimension.
            if(i != axis) {
                tmp.shape[j]    = op_in->shape[i];
                tmp.stride[j]   = op_in->stride[i];
                ++j;
            }
        }
        tmp.data = op_in->data;
       
        instr.status = BH_INST_PENDING;       // Copy the first element to the output.
        instr.opcode = BH_IDENTITY;
        instr.operand[0] = op_out;
        instr.operand[1] = &tmp;
        instr.operand[2] = NULL;

		bh_tstate state;
		bh_tstate_reset( &state, &instr );
		err = traverse_aa<T, T, identity_functor<T, T> >(&instr, &state);
        if (err != BH_SUCCESS) {
            return err;
        }

		bh_index stride = op_in->stride[axis];
		bh_index stride_bytes = stride * sizeof(T);

        instr.status = BH_INST_PENDING;       // Reduce over the 'axis' dimension.
        instr.opcode = opcode;                   // NB: the first element is already handled.
        instr.operand[0] = op_out;
        instr.operand[1] = op_out;
        instr.operand[2] = &tmp;

        tmp.start += stride;
		bh_tstate_reset( &state, &instr );

		void* out_start = state.start[0];
		void* tmp_start = state.start[2];
        
        for(i=1; i<nelements; ++i) {
						
            err = traverse_aaa<T, T, T, Instr>(&instr, &state);
            if (err != BH_SUCCESS) {
                return err;
            }

            tmp.start += stride;

			// Faster replacement of bh_tstate_reset
			tmp_start = (void*)(((char*)tmp_start) + stride_bytes);
            state.start[0] = out_start;
            state.start[1] = out_start;
            state.start[2] = tmp_start;
        }
    }

	return BH_SUCCESS;
}

bh_error bh_compute_reduce(bh_userfunc *arg, void* ve_arg)
{
    bh_reduce_type *a = (bh_reduce_type *) arg;   // Grab function arguments

    bh_opcode opcode = a->opcode;                    // Opcode
    bh_index axis    = a->axis;                      // The axis to reduce "around"

    bh_array *op_out = a->operand[0];                // Output operand
    bh_array *op_in  = a->operand[1];                // Input operand

                                                        //  Sanity checks.
    if (bh_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %lld is not a binary ufunc, hence it is not suitable for reduction.\n", (long long int)opcode);
        return BH_ERROR;
    }

	if (bh_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op[1] ) is null.\n");
        return BH_ERROR;
	}

    if (op_in->type != op_out->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return BH_ERROR;
    }
    
    if (bh_data_malloc(op_out) != BH_SUCCESS) {
        fprintf(stderr, "ERR: bh_compute_reduce; No memory for reduction-result.\n");
        return BH_OUT_OF_MEMORY;
    }

    long int poly = opcode + (op_in->type << 8);

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
        case BH_SUBTRACT + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, subtract_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, subtract_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, subtract_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, subtract_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, subtract_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, subtract_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, subtract_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, subtract_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, subtract_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, subtract_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, subtract_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, subtract_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_SUBTRACT + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, subtract_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
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
        case BH_DIVIDE + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, divide_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, divide_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, divide_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, divide_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, divide_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, divide_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, divide_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, divide_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, divide_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, divide_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, divide_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_DIVIDE + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, divide_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, power_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, power_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, power_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, power_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, power_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, power_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, power_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, power_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, power_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_POWER + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, power_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, greater_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, greater_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, greater_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, greater_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, greater_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, greater_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, greater_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, greater_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, greater_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, greater_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, greater_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, greater_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, greater_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, greater_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, greater_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, greater_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, greater_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, greater_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, greater_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, greater_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, greater_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_GREATER_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, greater_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, less_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, less_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, less_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, less_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, less_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, less_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, less_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, less_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, less_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, less_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LESS + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, less_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, less_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, less_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, less_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, less_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, less_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, less_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, less_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, less_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, less_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, less_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LESS_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, less_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, not_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, not_equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, not_equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, not_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, not_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, not_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, not_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, not_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, not_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, not_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, not_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, not_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_NOT_EQUAL + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, not_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_and_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_and_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_AND + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_or_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_or_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_OR + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_xor_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_xor_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LOGICAL_XOR + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
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
        case BH_LEFT_SHIFT + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, left_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, left_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, left_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, left_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, left_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, left_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, left_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_LEFT_SHIFT + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, left_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, right_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, right_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, right_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, right_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, right_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, right_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, right_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_RIGHT_SHIFT + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, right_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case BH_ARCTAN2 + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, arctan2_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_ARCTAN2 + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, arctan2_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, mod_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, mod_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, mod_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, mod_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, mod_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, mod_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, mod_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, mod_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, mod_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case BH_MOD + (BH_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, mod_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:
            
            return BH_ERROR;

    }

}

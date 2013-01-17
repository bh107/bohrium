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

        return CPHVB_SUCCESS;

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
       
        instr.status = CPHVB_INST_PENDING;       // Copy the first element to the output.
        instr.opcode = CPHVB_IDENTITY;
        instr.operand[0] = op_out;
        instr.operand[1] = &tmp;
        instr.operand[2] = NULL;

        // TODO: use traverse directly
        //err = traverse_aa<T, T, Instr>( instr );// execute the pseudo-instruction
        //err = bh_compute_apply( &instr );
        err = bh_compute_apply_naive( &instr );
        if (err != CPHVB_SUCCESS) {
            return err;
        }
        tmp.start += stride;

        instr.status = CPHVB_INST_PENDING;       // Reduce over the 'axis' dimension.
        instr.opcode = opcode;                   // NB: the first element is already handled.
        instr.operand[0] = op_out;
        instr.operand[1] = op_out;
        instr.operand[2] = &tmp;
        
        for(i=1; i<nelements; ++i) {
            // TODO: use traverse directly
            //err = traverse_aaa<T, T, T, Instr>( instr );
            err = bh_compute_apply_naive( &instr );
            if (err != CPHVB_SUCCESS) {
                return err;
            }
            tmp.start += stride;
        }

        return CPHVB_SUCCESS;
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
        return CPHVB_ERROR;
    }

	if (bh_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op[1] ) is null.\n");
        return CPHVB_ERROR;
	}

    if (op_in->type != op_out->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return CPHVB_ERROR;
    }
    
    if (bh_data_malloc(op_out) != CPHVB_SUCCESS) {
        fprintf(stderr, "ERR: bh_compute_reduce; No memory for reduction-result.\n");
        return CPHVB_OUT_OF_MEMORY;
    }

    long int poly = opcode + (op_in->type << 8);

    switch(poly) {
    
        case CPHVB_ADD + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, add_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, add_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, add_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, add_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, add_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, add_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, add_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, add_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, add_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, add_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, add_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, add_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, add_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, subtract_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, subtract_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, subtract_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, subtract_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, subtract_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, subtract_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, subtract_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, subtract_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, subtract_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, subtract_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, subtract_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, subtract_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, subtract_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, multiply_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, multiply_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, multiply_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, multiply_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, multiply_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, multiply_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, multiply_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, multiply_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, multiply_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, multiply_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, multiply_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, multiply_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, multiply_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, divide_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, divide_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, divide_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, divide_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, divide_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, divide_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, divide_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, divide_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, divide_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, divide_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, divide_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, divide_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, power_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, power_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, power_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, power_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, power_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, power_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, power_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, power_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, power_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, power_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, greater_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, greater_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, greater_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, greater_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, greater_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, greater_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, greater_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, greater_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, greater_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, greater_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, greater_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, greater_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, greater_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, greater_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, greater_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, greater_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, greater_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, greater_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, greater_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, greater_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, greater_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, greater_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, less_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, less_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, less_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, less_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, less_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, less_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, less_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, less_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, less_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, less_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, less_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, less_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, less_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, less_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, less_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, less_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, less_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, less_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, less_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, less_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, less_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, less_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, not_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any_naive<std::complex<double>, not_equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any_naive<std::complex<float>, not_equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, not_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, not_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, not_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, not_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, not_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, not_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, not_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, not_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, not_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, not_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_and_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_and_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_or_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_or_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, logical_xor_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, logical_xor_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, logical_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, logical_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, logical_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, logical_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, logical_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, logical_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, logical_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, logical_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, maximum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, maximum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, maximum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, maximum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, maximum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, maximum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, maximum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, maximum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, maximum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, maximum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, maximum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, minimum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, minimum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, minimum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, minimum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, minimum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, minimum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, minimum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, minimum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, minimum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, minimum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, minimum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any_naive<bh_bool, bitwise_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, bitwise_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, bitwise_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, bitwise_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, bitwise_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, bitwise_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, bitwise_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, bitwise_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, bitwise_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, left_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, left_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, left_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, left_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, left_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, left_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, left_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, left_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, right_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, right_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, right_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, right_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, right_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, right_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, right_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, right_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, arctan2_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, arctan2_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any_naive<bh_float32, mod_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any_naive<bh_float64, mod_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any_naive<bh_int16, mod_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any_naive<bh_int32, mod_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any_naive<bh_int64, mod_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any_naive<bh_int8, mod_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any_naive<bh_uint16, mod_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any_naive<bh_uint32, mod_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any_naive<bh_uint64, mod_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any_naive<bh_uint8, mod_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:
            
            return CPHVB_ERROR;

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
       
        instr.status = CPHVB_INST_PENDING;       // Copy the first element to the output.
        instr.opcode = CPHVB_IDENTITY;
        instr.operand[0] = op_out;
        instr.operand[1] = &tmp;
        instr.operand[2] = NULL;

		bh_tstate state;
		bh_tstate_reset( &state, &instr );
		err = traverse_aa<T, T, identity_functor<T, T> >(&instr, &state);
        if (err != CPHVB_SUCCESS) {
            return err;
        }

		bh_index stride = op_in->stride[axis];
		bh_index stride_bytes = stride * sizeof(T);

        instr.status = CPHVB_INST_PENDING;       // Reduce over the 'axis' dimension.
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
            if (err != CPHVB_SUCCESS) {
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

	return CPHVB_SUCCESS;
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
        return CPHVB_ERROR;
    }

	if (bh_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: bh_compute_reduce; input-operand ( op[1] ) is null.\n");
        return CPHVB_ERROR;
	}

    if (op_in->type != op_out->type) {
        fprintf(stderr, "ERR: bh_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return CPHVB_ERROR;
    }
    
    if (bh_data_malloc(op_out) != CPHVB_SUCCESS) {
        fprintf(stderr, "ERR: bh_compute_reduce; No memory for reduction-result.\n");
        return CPHVB_OUT_OF_MEMORY;
    }

    long int poly = opcode + (op_in->type << 8);

    switch(poly) {
    
        case CPHVB_ADD + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, add_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, add_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, add_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, add_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, add_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, add_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, add_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, add_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, add_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, add_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, add_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, add_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_ADD + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, add_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, subtract_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, subtract_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, subtract_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, subtract_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, subtract_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, subtract_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, subtract_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, subtract_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, subtract_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, subtract_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, subtract_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, subtract_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, subtract_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, multiply_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, multiply_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, multiply_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, multiply_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, multiply_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, multiply_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, multiply_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, multiply_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, multiply_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, multiply_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, multiply_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, multiply_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, multiply_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, divide_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, divide_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, divide_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, divide_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, divide_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, divide_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, divide_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, divide_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, divide_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, divide_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, divide_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, divide_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, power_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, power_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, power_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, power_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, power_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, power_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, power_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, power_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, power_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_POWER + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, power_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, greater_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, greater_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, greater_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, greater_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, greater_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, greater_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, greater_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, greater_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, greater_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, greater_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, greater_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, greater_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, greater_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, greater_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, greater_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, greater_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, greater_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, greater_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, greater_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, greater_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, greater_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, greater_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, less_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, less_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, less_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, less_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, less_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, less_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, less_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, less_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, less_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, less_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, less_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, less_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, less_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, less_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, less_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, less_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, less_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, less_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, less_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, less_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, less_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, less_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, not_equal_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX128 << 8):
            return bh_compute_reduce_any<std::complex<double>, not_equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX64 << 8):
            return bh_compute_reduce_any<std::complex<float>, not_equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, not_equal_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, not_equal_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, not_equal_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, not_equal_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, not_equal_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, not_equal_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, not_equal_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, not_equal_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, not_equal_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, not_equal_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_and_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_and_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_or_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_or_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, logical_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, logical_xor_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, logical_xor_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, logical_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, logical_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, logical_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, logical_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, logical_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, logical_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, logical_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, logical_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, maximum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, maximum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, maximum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, maximum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, maximum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, maximum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, maximum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, maximum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, maximum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, maximum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, maximum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, minimum_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, minimum_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, minimum_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, minimum_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, minimum_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, minimum_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, minimum_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, minimum_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, minimum_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, minimum_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, minimum_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_and_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_and_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_and_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_and_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_and_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_and_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_and_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_and_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_and_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_or_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_or_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_or_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_or_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_or_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_or_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_or_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_or_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_or_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8):
            return bh_compute_reduce_any<bh_bool, bitwise_xor_functor<bh_bool, bh_bool, bh_bool > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, bitwise_xor_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, bitwise_xor_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, bitwise_xor_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, bitwise_xor_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, bitwise_xor_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, bitwise_xor_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, bitwise_xor_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, bitwise_xor_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, left_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, left_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, left_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, left_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, left_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, left_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, left_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, left_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, right_shift_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, right_shift_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, right_shift_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, right_shift_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, right_shift_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, right_shift_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, right_shift_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, right_shift_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, arctan2_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, arctan2_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8):
            return bh_compute_reduce_any<bh_float32, mod_functor<bh_float32, bh_float32, bh_float32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8):
            return bh_compute_reduce_any<bh_float64, mod_functor<bh_float64, bh_float64, bh_float64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT16 << 8):
            return bh_compute_reduce_any<bh_int16, mod_functor<bh_int16, bh_int16, bh_int16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT32 << 8):
            return bh_compute_reduce_any<bh_int32, mod_functor<bh_int32, bh_int32, bh_int32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT64 << 8):
            return bh_compute_reduce_any<bh_int64, mod_functor<bh_int64, bh_int64, bh_int64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_INT8 << 8):
            return bh_compute_reduce_any<bh_int8, mod_functor<bh_int8, bh_int8, bh_int8 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT16 << 8):
            return bh_compute_reduce_any<bh_uint16, mod_functor<bh_uint16, bh_uint16, bh_uint16 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT32 << 8):
            return bh_compute_reduce_any<bh_uint32, mod_functor<bh_uint32, bh_uint32, bh_uint32 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT64 << 8):
            return bh_compute_reduce_any<bh_uint64, mod_functor<bh_uint64, bh_uint64, bh_uint64 > >( op_out, op_in, axis, opcode );
        case CPHVB_MOD + (CPHVB_UINT8 << 8):
            return bh_compute_reduce_any<bh_uint8, mod_functor<bh_uint8, bh_uint8, bh_uint8 > >( op_out, op_in, axis, opcode );

        default:
            
            return CPHVB_ERROR;

    }

}

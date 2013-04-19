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

        data_out += op_out->start;
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
       
        instr.opcode = BH_IDENTITY;       // Copy the first element to the output.
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

        instr.opcode = opcode;                   // Reduce over the 'axis' dimension.
        instr.operand[0] = op_out;               // NB: the first element is already handled.
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
       
        instr.opcode = BH_IDENTITY;					// Copy the first element to the output.
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

        instr.opcode = opcode;                // Reduce over the 'axis' dimension.
        instr.operand[0] = op_out;			  // NB: the first element is already handled.
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

        default:
            
            return BH_ERROR;

    }

}

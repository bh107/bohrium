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
#include <cphvb.h>
#include <cphvb_compute.h>
#include <cphvb_compute.h>
#include "functors.hpp"
#include <complex>
#include "traverser.hpp"

typedef char BYTE;


/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param opt_out The output scalar storage array
 *  @param opt_in The array to aggregate
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename Instr>
cphvb_error traverse_sa( cphvb_array* op_out, cphvb_array* op_in ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointer to start of data elements
	BYTE* d1;

    size_t elsize1 = sizeof(T1);				// We use the size for explicit multiplication

	T1 value;									// Local store with data
	
    d0 = (BYTE*) cphvb_base_array(op_out)->data;
    d1 = (BYTE*) cphvb_base_array(op_in)->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

	d1 += op_in->start * elsize1;				// Compute offsets
	value = *((T1*)d1);
		
	if (op_in->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride1 = op_in->stride[0] * elsize1;
		
		cphvb_index total_ops = op_in->shape[0];
		
		d1 += stride1;
		total_ops--;

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_SSA(opcode_func, fulls, remainder, d1, stride1, &value);
	}
	else if(op_in->ndim == 2)
	{
		cphvb_index ops_outer = op_in->shape[0];
		cphvb_index ops_inner = op_in->shape[1];
		
		// Basic 2D loop with unrolling
		cphvb_index outer_stride1 = op_in->stride[0] * elsize1;
		cphvb_index inner_stride1 = op_in->stride[1] * elsize1;
		outer_stride1 -= inner_stride1 * op_in->shape[1];

		cphvb_index remainder = (ops_inner - 1) % 4;
		cphvb_index fulls = (ops_inner - 1) / 4;

		// Skip the first element, as that is copied
		d1 += inner_stride1;
		INNER_LOOP_SSA(opcode_func, fulls, remainder, d1, inner_stride1, &value);

		remainder = ops_inner % 4;
		fulls = ops_inner / 4;

		for (i = 1; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_SSA(opcode_func, fulls, remainder, d1, inner_stride1, &value);
			d1 += outer_stride1;
		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = op_in->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= op_in->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = op_in->shape[dim_index0];
		cphvb_index ops_inner = op_in->shape[dim_index1];
		cphvb_index ops_inner_inner = op_in->shape[dim_index2];

		cphvb_index outer_stride1 = op_in->stride[dim_index0] * elsize1;
		cphvb_index inner_stride1 = op_in->stride[dim_index1] * elsize1;
		cphvb_index inner_inner_stride1 = op_in->stride[dim_index2] * elsize1;

		outer_stride1 -= inner_stride1 * op_in->shape[dim_index1];
		inner_stride1 -= inner_inner_stride1 * op_in->shape[dim_index2];

		cphvb_index remainder = (ops_inner_inner - 1) % 4;
		cphvb_index fulls = (ops_inner_inner - 1) / 4;

		BYTE* d1_orig = d1;

		// Run the first inner loop without the first element
		d1 += inner_inner_stride1;
		INNER_LOOP_SSA(opcode_func, fulls, remainder, d1, inner_inner_stride1, &value);
		d1 += inner_stride1; //Patch to start at next entry

		// Prepare for the following loops
		remainder = ops_inner_inner % 4;
		fulls = ops_inner_inner / 4;
		cphvb_index j_start = 1;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = j_start; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_SSA(opcode_func, fulls, remainder, d1, inner_inner_stride1, &value);
					d1 += inner_stride1;
				}

				d1 += outer_stride1;
				j_start = 0;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d1_orig += (op_in->stride[p] * elsize1);

				while (++counters[p] == op_in->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d1_orig += ((op_in->stride[p-1]) - (op_in->shape[p] * op_in->stride[p])) * elsize1;

					counters[p] = 0;
					p--;
				}
				
				d1 = d1_orig;
			}
		}		
	}

	*(((T0*)d0) + op_out->start) = (T1)value;

    return CPHVB_SUCCESS;

}

cphvb_error cphvb_compute_aggregate(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_aggregate_type *a = (cphvb_aggregate_type *) arg;   // Grab function arguments

    cphvb_opcode opcode = a->opcode;                    // Opcode

    cphvb_array *op_out = a->operand[0];                // Output operand
    cphvb_array *op_in  = a->operand[1];                // Input operand

                                                        //  Sanity checks.
    if (cphvb_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %lld is not a binary ufunc, hence it is not suitable for reduction.\n", (long long int)opcode);
        return CPHVB_ERROR;
    }

	if (cphvb_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: cphvb_compute_aggregate; input-operand ( op[1] ) is null.\n");
        return CPHVB_ERROR;
	}

    if (op_in->type != op_out->type) {
        fprintf(stderr, "ERR: cphvb_compute_aggregate; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n");
        return CPHVB_ERROR;
    }
    
    if (cphvb_data_malloc(op_out) != CPHVB_SUCCESS) {
        fprintf(stderr, "ERR: cphvb_compute_aggregate; No memory for reduction-result.\n");
        return CPHVB_OUT_OF_MEMORY;
    }
    
    if (op_out->ndim != 1 || op_out->shape[0] != 1) {
        fprintf(stderr, "ERR: cphvb_compute_aggregate; output-operand ( op[0] ) is not a scalar.\n");
        return CPHVB_ERROR;
    }

    long int poly = opcode + (op_in->type << 8);

    switch(poly) {
    
        case CPHVB_ADD + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, add_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, add_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, add_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, add_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, add_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, add_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, add_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, add_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, add_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, add_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, add_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, add_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_ADD + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, add_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, subtract_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, subtract_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, subtract_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, subtract_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, subtract_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, subtract_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, subtract_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, subtract_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, subtract_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, subtract_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, subtract_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, subtract_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_SUBTRACT + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, subtract_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, multiply_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, multiply_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, multiply_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, multiply_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, multiply_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, multiply_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, multiply_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, multiply_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, multiply_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, multiply_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, multiply_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, multiply_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_MULTIPLY + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, multiply_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, divide_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, divide_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, divide_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, divide_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, divide_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, divide_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, divide_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, divide_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, divide_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, divide_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, divide_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_DIVIDE + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, divide_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, power_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, power_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, power_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, power_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, power_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, power_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, power_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, power_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, power_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_POWER + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, power_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, greater_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, greater_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, greater_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, greater_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, greater_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, greater_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, greater_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, greater_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, greater_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, greater_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_GREATER + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, greater_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, greater_equal_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, greater_equal_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, greater_equal_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, greater_equal_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, greater_equal_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, greater_equal_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, greater_equal_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, greater_equal_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, greater_equal_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, greater_equal_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_GREATER_EQUAL + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, greater_equal_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, less_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, less_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, less_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, less_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, less_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, less_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, less_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, less_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, less_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, less_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LESS + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, less_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, less_equal_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, less_equal_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, less_equal_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, less_equal_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, less_equal_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, less_equal_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, less_equal_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, less_equal_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, less_equal_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, less_equal_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LESS_EQUAL + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, less_equal_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, equal_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, equal_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, equal_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, equal_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, equal_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, equal_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, equal_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, equal_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, equal_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, equal_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_EQUAL + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, equal_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, not_equal_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX128 << 8):
            return traverse_sa<std::complex<double>, std::complex<double>, not_equal_functor<std::complex<double>, std::complex<double>, std::complex<double> > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_COMPLEX64 << 8):
            return traverse_sa<std::complex<float>, std::complex<float>, not_equal_functor<std::complex<float>, std::complex<float>, std::complex<float> > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, not_equal_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, not_equal_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, not_equal_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, not_equal_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, not_equal_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, not_equal_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, not_equal_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, not_equal_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, not_equal_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_NOT_EQUAL + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, not_equal_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, logical_and_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, logical_and_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, logical_and_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, logical_and_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, logical_and_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, logical_and_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, logical_and_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, logical_and_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, logical_and_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, logical_and_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LOGICAL_AND + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, logical_and_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, logical_or_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, logical_or_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, logical_or_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, logical_or_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, logical_or_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, logical_or_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, logical_or_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, logical_or_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, logical_or_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, logical_or_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LOGICAL_OR + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, logical_or_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, logical_xor_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, logical_xor_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, logical_xor_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, logical_xor_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, logical_xor_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, logical_xor_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, logical_xor_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, logical_xor_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, logical_xor_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, logical_xor_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LOGICAL_XOR + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, logical_xor_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, maximum_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, maximum_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, maximum_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, maximum_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, maximum_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, maximum_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, maximum_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, maximum_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, maximum_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, maximum_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_MAXIMUM + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, maximum_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, minimum_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, minimum_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, minimum_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, minimum_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, minimum_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, minimum_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, minimum_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, minimum_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, minimum_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, minimum_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_MINIMUM + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, minimum_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, bitwise_and_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, bitwise_and_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, bitwise_and_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, bitwise_and_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, bitwise_and_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, bitwise_and_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, bitwise_and_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, bitwise_and_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_BITWISE_AND + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, bitwise_and_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, bitwise_or_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, bitwise_or_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, bitwise_or_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, bitwise_or_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, bitwise_or_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, bitwise_or_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, bitwise_or_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, bitwise_or_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_BITWISE_OR + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, bitwise_or_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_BOOL << 8):
            return traverse_sa<cphvb_bool, cphvb_bool, bitwise_xor_functor<cphvb_bool, cphvb_bool, cphvb_bool > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, bitwise_xor_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, bitwise_xor_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, bitwise_xor_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, bitwise_xor_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, bitwise_xor_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, bitwise_xor_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, bitwise_xor_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_BITWISE_XOR + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, bitwise_xor_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, left_shift_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, left_shift_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, left_shift_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, left_shift_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, left_shift_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, left_shift_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, left_shift_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_LEFT_SHIFT + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, left_shift_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, right_shift_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, right_shift_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, right_shift_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, right_shift_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, right_shift_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, right_shift_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, right_shift_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_RIGHT_SHIFT + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, right_shift_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, arctan2_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_ARCTAN2 + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, arctan2_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_FLOAT32 << 8):
            return traverse_sa<cphvb_float32, cphvb_float32, mod_functor<cphvb_float32, cphvb_float32, cphvb_float32 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_FLOAT64 << 8):
            return traverse_sa<cphvb_float64, cphvb_float64, mod_functor<cphvb_float64, cphvb_float64, cphvb_float64 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_INT16 << 8):
            return traverse_sa<cphvb_int16, cphvb_int16, mod_functor<cphvb_int16, cphvb_int16, cphvb_int16 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_INT32 << 8):
            return traverse_sa<cphvb_int32, cphvb_int32, mod_functor<cphvb_int32, cphvb_int32, cphvb_int32 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_INT64 << 8):
            return traverse_sa<cphvb_int64, cphvb_int64, mod_functor<cphvb_int64, cphvb_int64, cphvb_int64 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_INT8 << 8):
            return traverse_sa<cphvb_int8, cphvb_int8, mod_functor<cphvb_int8, cphvb_int8, cphvb_int8 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_UINT16 << 8):
            return traverse_sa<cphvb_uint16, cphvb_uint16, mod_functor<cphvb_uint16, cphvb_uint16, cphvb_uint16 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_UINT32 << 8):
            return traverse_sa<cphvb_uint32, cphvb_uint32, mod_functor<cphvb_uint32, cphvb_uint32, cphvb_uint32 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_UINT64 << 8):
            return traverse_sa<cphvb_uint64, cphvb_uint64, mod_functor<cphvb_uint64, cphvb_uint64, cphvb_uint64 > >( op_out, op_in );
        case CPHVB_MOD + (CPHVB_UINT8 << 8):
            return traverse_sa<cphvb_uint8, cphvb_uint8, mod_functor<cphvb_uint8, cphvb_uint8, cphvb_uint8 > >( op_out, op_in );

        default:
            
            return CPHVB_ERROR;

    }

}

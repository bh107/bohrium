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
#include "bh_matmul.h"
#include <complex>

template <typename Type> bh_error do_matmul(bh_view *A, bh_view *B, bh_view *C){

    Type* A_data;
    Type* B_data;
    Type* C_data;

    bh_data_get(A, (bh_data_ptr*) &A_data);
    bh_data_get(B, (bh_data_ptr*) &B_data);
    bh_data_get(C, (bh_data_ptr*) &C_data);

    bh_intp M = A->shape[0];
    bh_intp N = B->shape[1];
    bh_intp K = A->shape[1];

    for(bh_intp i = 0; i < M; i++){
        for(bh_intp j = 0; j < N; j++){
            C_data[C->start + i*C->stride[0]+j*C->stride[1]] = 0;
            for(bh_intp k = 0; k < K; k++){
                C_data[C->start + i*C->stride[0]+j*C->stride[1]] += \
                         A_data[A->start + i*A->stride[0]+k*A->stride[1]] * \
                           B_data[B->start + k*B->stride[0]+j*B->stride[1]];
            }
        }
    }

    return BH_SUCCESS;
}

/* Implements matrix multiplication */
bh_error bh_matmul(bh_instruction *instr, void* arg)
{
    bh_view *C = &instr->operand[0];
    bh_view *A = &instr->operand[1];
    bh_view *B = &instr->operand[2];

    //Make sure that the arrays memory are allocated.
    if(bh_data_malloc(A->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if(bh_data_malloc(B->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if(bh_data_malloc(C->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;

    switch (bh_base_array(C)->type)
    {
    	case BH_INT8:
		    return do_matmul<bh_int8>(A, B, C);
    	case BH_INT16:
		    return do_matmul<bh_int16>(A, B, C);
    	case BH_INT32:
		    return do_matmul<bh_int32>(A, B, C);
    	case BH_INT64:
		    return do_matmul<bh_int64>(A, B, C);
        case BH_UINT8:
		    return do_matmul<bh_uint8>(A, B, C);
    	case BH_UINT16:
		    return do_matmul<bh_uint16>(A, B, C);
    	case BH_UINT32:
	            return do_matmul<bh_uint32>(A, B, C);
    	case BH_UINT64:
		    return do_matmul<bh_uint64>(A, B, C);
    	case BH_FLOAT32:
		    return do_matmul<bh_float32>(A, B, C);
    	case BH_FLOAT64:
		    return do_matmul<bh_float64>(A, B, C);
        case BH_COMPLEX64:
            return do_matmul<std::complex<float> >(A, B, C);
        case BH_COMPLEX128:
            return do_matmul<std::complex<double> >(A, B, C);
        default:
            return BH_TYPE_NOT_SUPPORTED;
    }
}


#include <cphvb.h>
#include <cphvb_compute.h>


template <typename Type> cphvb_error do_matmul(cphvb_array *A, cphvb_array *B, cphvb_array *C){

    if(cphvb_data_malloc(C) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    

    Type* A_data;
    Type* B_data;
    Type* C_data;

    cphvb_data_get(A, (cphvb_data_ptr*) &A_data);
    cphvb_data_get(B, (cphvb_data_ptr*) &B_data);
    cphvb_data_get(C, (cphvb_data_ptr*) &C_data);

    cphvb_intp M = A->shape[0];
    cphvb_intp N = B->shape[1];
    cphvb_intp K = A->shape[1];

    for(cphvb_intp i = 0; i < M; i++){
        for(cphvb_intp j = 0; j < N; j++){
            C_data[C->start + i*C->stride[0]+j*C->stride[1]] = 0;
            for(cphvb_intp k = 0; k < K; k++){
                C_data[C->start + i*C->stride[0]+j*C->stride[1]] += \
                         A_data[A->start + i*A->stride[0]+k*A->stride[1]] * \
                           B_data[B->start + k*B->stride[0]+j*B->stride[1]];
            }
        }	
    }
    
    return CPHVB_SUCCESS;
}



/**
 * cphvb_compute_matmul
 *
 * Implementation of the user-defined funtion "matmul".
 * Note that we follow the function signature defined by cphvb_userfunc_impl.
 *
 */
cphvb_error cphvb_compute_matmul(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_matmul_type *m_arg = (cphvb_matmul_type *) arg;
    cphvb_array *C = m_arg->operand[0];
    cphvb_array *A = m_arg->operand[1];
    cphvb_array *B = m_arg->operand[2];

    //Make sure that the arrays memory are allocated.
    if(cphvb_data_malloc(A) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 
    if(cphvb_data_malloc(B) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 
    if(cphvb_data_malloc(C) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 

    switch (C->type)
    {
    	case CPHVB_INT8:
		    return do_matmul<cphvb_int8>(A, B, C);
    	case CPHVB_INT16:
		    return do_matmul<cphvb_int16>(A, B, C);
    	case CPHVB_INT32:
		    return do_matmul<cphvb_int32>(A, B, C);
    	case CPHVB_INT64:
		    return do_matmul<cphvb_int64>(A, B, C);
        case CPHVB_UINT8:
		    return do_matmul<cphvb_uint8>(A, B, C);
    	case CPHVB_UINT16:
		    return do_matmul<cphvb_uint16>(A, B, C);
    	case CPHVB_UINT32:
	        return do_matmul<cphvb_uint32>(A, B, C);
    	case CPHVB_UINT64:
		    return do_matmul<cphvb_uint64>(A, B, C);
    	case CPHVB_FLOAT32:
		    return do_matmul<cphvb_float32>(A, B, C);
    	case CPHVB_FLOAT64:
		    return do_matmul<cphvb_float64>(A, B, C);
        default:
            return CPHVB_TYPE_NOT_SUPPORTED;

	}

}




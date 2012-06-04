/*
 * Copyright 2012 Andreas Thorning <thorning@diku.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cphvb.h>
#include <cblas.h>


cphvb_error do_matmul_float32(cphvb_array *A, cphvb_array *B, cphvb_array *C)
{
    cphvb_float32* A_data;
    cphvb_float32* B_data;
    cphvb_float32* C_data;

    cphvb_data_get(A, (cphvb_data_ptr*) &A_data);
    cphvb_data_get(B, (cphvb_data_ptr*) &B_data);
    cphvb_data_get(C, (cphvb_data_ptr*) &C_data);

    A_data += A->start;
    B_data += B->start;
    C_data += C->start;

    cphvb_intp M = A->shape[0];
    cphvb_intp N = B->shape[1];
    cphvb_intp K = A->shape[1];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A_data, A->stride[0], B_data, B->stride[0], 0.0, C_data, C->stride[0]);
    
    return CPHVB_SUCCESS;
}

cphvb_error do_matmul_float64(cphvb_array *A, cphvb_array *B, cphvb_array *C)
{

    cphvb_float64* A_data;
    cphvb_float64* B_data;
    cphvb_float64* C_data;

    cphvb_data_get(A, (cphvb_data_ptr*) &A_data);
    cphvb_data_get(B, (cphvb_data_ptr*) &B_data);
    cphvb_data_get(C, (cphvb_data_ptr*) &C_data);

    A_data += A->start;
    B_data += B->start;
    C_data += C->start;

    cphvb_intp M = A->shape[0];
    cphvb_intp N = B->shape[1];
    cphvb_intp K = A->shape[1];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A_data, A->stride[0], B_data, B->stride[0], 0.0, C_data, C->stride[0]);
    
    return CPHVB_SUCCESS;
}

cphvb_error cphvb_matmul(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_matmul_type *m_arg = (cphvb_matmul_type *) arg;
    cphvb_array *C = m_arg->operand[0];
    cphvb_array *A = m_arg->operand[1];
    cphvb_array *B = m_arg->operand[2];

    //We only supports data that are sequential in the row
    if(A->stride[1] != 1 || B->stride[1] != 1 || C->stride[1] != 1)
        goto failback;

    if(cphvb_data_malloc(A) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    
    if(cphvb_data_malloc(B) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    
    if(cphvb_data_malloc(C) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    

    switch (C->type)
    {
    	case CPHVB_FLOAT32:
	    	return do_matmul_float32(A, B, C);
    	case CPHVB_FLOAT64:
	    	return do_matmul_float64(A, B, C);
    	default:
            goto failback;
	}

    //The naive failback implementation that supports views 
    //and all data types
    failback:
        return cphvb_compute_matmul(arg, ve_arg);
    return CPHVB_ERROR;
}

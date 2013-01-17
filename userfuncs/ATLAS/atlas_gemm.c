/*
 * Copyright 2012 Andreas Thorning <thorning@diku.dk>
 *
 * This file is part of Bohrium.
 *
 * Bohrium is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Bohrium is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Bohrium. If not, see <http://www.gnu.org/licenses/>.
 */
#include <bh.h>
extern "C" {
#include <cblas.h>
}


bh_error do_matmul_float32(bh_array *A, bh_array *B, bh_array *C)
{
    bh_float32* A_data;
    bh_float32* B_data;
    bh_float32* C_data;

    bh_data_get(A, (bh_data_ptr*) &A_data);
    bh_data_get(B, (bh_data_ptr*) &B_data);
    bh_data_get(C, (bh_data_ptr*) &C_data);

    A_data += A->start;
    B_data += B->start;
    C_data += C->start;

    bh_intp M = A->shape[0];
    bh_intp N = B->shape[1];
    bh_intp K = A->shape[1];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A_data, A->stride[0], B_data, B->stride[0], 0.0, C_data, C->stride[0]);
    
    return CPHVB_SUCCESS;
}

bh_error do_matmul_float64(bh_array *A, bh_array *B, bh_array *C)
{

    bh_float64* A_data;
    bh_float64* B_data;
    bh_float64* C_data;

    bh_data_get(A, (bh_data_ptr*) &A_data);
    bh_data_get(B, (bh_data_ptr*) &B_data);
    bh_data_get(C, (bh_data_ptr*) &C_data);

    A_data += A->start;
    B_data += B->start;
    C_data += C->start;

    bh_intp M = A->shape[0];
    bh_intp N = B->shape[1];
    bh_intp K = A->shape[1];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A_data, A->stride[0], B_data, B->stride[0], 0.0, C_data, C->stride[0]);
    
    return CPHVB_SUCCESS;
}

bh_error bh_matmul(bh_userfunc *arg, void* ve_arg)
{   
    bh_matmul_type *m_arg = (bh_matmul_type *) arg;
    bh_array *C = m_arg->operand[0];
    bh_array *A = m_arg->operand[1];
    bh_array *B = m_arg->operand[2];

    //We only supports data that are sequential in the row
    if(A->stride[1] != 1 || B->stride[1] != 1 || C->stride[1] != 1)
        goto failback;

    if(bh_data_malloc(A) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    
    if(bh_data_malloc(B) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;    
    if(bh_data_malloc(C) != CPHVB_SUCCESS)
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
        return bh_compute_matmul(arg, ve_arg);
    return CPHVB_ERROR;
}

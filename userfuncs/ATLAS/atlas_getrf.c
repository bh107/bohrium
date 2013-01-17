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
#include <cassert>
extern "C" {
#include <clapack.h>
#include <cblas.h>
}

void transpose_float32(bh_float32* A, int n, int row_stride){
  int i, j;
  #pragma omp parallel for
  for(j = 0; j < n-1; j++){
    for(i = j+1; i < n; i++){
       bh_float32 temp = A[i + j*row_stride];
       A[i + j*row_stride] = A[j + i*row_stride];
       A[j + i*row_stride] = temp;
    }
  }
}

void transpose_float64(bh_float64* A, int n, int row_stride){
  int i, j;
  #pragma omp parallel for
  for(i = 0; i < n-1; i++){
    for(j = i+1; j < n; j++){
       bh_float64 temp = A[i + j*row_stride];
       A[i + j*row_stride] = A[j + i*row_stride];
       A[j + i*row_stride] = temp;
    }
  }
}

bh_error do_lu_float32(bh_array *A, bh_array *P){

    bh_float32* A_data;
    bh_int32* P_data;

    bh_data_get(A, (bh_data_ptr*) &A_data);
    bh_data_get(P, (bh_data_ptr*) &P_data);

    A_data += A->start;
    P_data += P->start;
    
    bh_intp N = A->shape[0];
    
    //unfortinatly, atlas define row_major lu with column povoting, meaning we need to transpose the matrix
    //to get same behavior as scipy
    transpose_float32(A_data, N, A->stride[0]);
    clapack_sgetrf(CblasRowMajor, N, N, A_data, A->stride[0], P_data);
    transpose_float32(A_data, N, A->stride[0]);
    
    return CPHVB_SUCCESS;
}

bh_error do_lu_float64(bh_array *A, bh_array *P){

    bh_float64* A_data;
    bh_int32* P_data;

    bh_data_get(A, (bh_data_ptr*) &A_data);
    bh_data_get(P, (bh_data_ptr*) &P_data);

    A_data += A->start;
    P_data += P->start;
    
    bh_intp N = A->shape[0];
    
    //unfortinatly, atlas define row_major lu with column povoting, meaning we need to transpose the matrix
    //to get same behavior as scipy
    transpose_float64(A_data, N, A->stride[0]);
    clapack_dgetrf(CblasRowMajor, N, N, A_data, A->stride[0], P_data);
    transpose_float64(A_data, N, A->stride[0]);
    
    return CPHVB_SUCCESS;
}

bh_error bh_lu( bh_userfunc *arg, void* ve_arg)
{
    bh_lu_type *m_arg = (bh_lu_type *) arg;
    bh_array *A = m_arg->operand[0];
    bh_array *P = m_arg->operand[1];
    
    if(bh_data_malloc(A) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
        
    if(bh_data_malloc(P) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;

    //A needs to be row major, P needs to be continuous. Should be no problem, as both arrays are just created
    if(A->stride[1] != 1 || P->stride[0] != 1)
        return CPHVB_ERROR;
    
    switch (A->type)
    {
    	case CPHVB_FLOAT32:
	    	return do_lu_float32(A, P);
    	case CPHVB_FLOAT64:
	    	return do_lu_float64(A, P);
    	default:
            return CPHVB_ERROR;
	}   
    
}

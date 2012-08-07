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
#include <fftw3.h>
#include <omp.h>

/* ONE DIMENSIONAL TRANSFORMATIONS START */

cphvb_error do_fft_complex64(cphvb_array* in, cphvb_array* out){

  cphvb_complex64* in_data;
  cphvb_complex64* out_data;
  
  cphvb_data_get(in, (cphvb_data_ptr*) &in_data);
  cphvb_data_get(out, (cphvb_data_ptr*) &out_data);
  

  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());
  fftwf_plan p;
  if ( in->ndim == 1)
    p = fftwf_plan_dft_1d(in->shape[0], (fftwf_complex*) in_data, (fftwf_complex*) out_data, FFTW_FORWARD, FFTW_ESTIMATE);
  else if ( in->ndim == 2){
    p = fftwf_plan_many_dft(1, (int*)&in->shape[1], in->shape[0],
                                  (fftwf_complex*) in_data, 0,
                                  in->stride[1], in->stride[0],
                                  (fftwf_complex*) out_data, 0,
                                  out->stride[1], out->stride[0],
                                  FFTW_FORWARD, FFTW_ESTIMATE);
     }
  else
    return CPHVB_ERROR;
  fftwf_execute(p);
  fftwf_destroy_plan(p);
  fftwf_cleanup_threads();
  
  return CPHVB_SUCCESS;
}

cphvb_error do_fft_complex128(cphvb_array* in, cphvb_array* out){

  cphvb_complex128* in_data;
  cphvb_complex128* out_data;
  
  cphvb_data_get(in, (cphvb_data_ptr*) &in_data);
  cphvb_data_get(out, (cphvb_data_ptr*) &out_data);
  

  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fftw_plan p;
  int n = in->ndim == 1 ? in->shape[0] : in->shape[1];
  int how_many = in->ndim == 1 ? 1 : in->shape[0];
  int in_stride = in->ndim == 1 ? in->stride[0] : in->stride[1];
  int in_dist = in->ndim == 1 ? 0 : in->stride[0];
  int out_stride = out->ndim == 1 ? out->stride[0] : out->stride[1];
  int out_dist = out->ndim == 1 ? 0 : out->stride[0];
  p = fftw_plan_many_dft(1, &n, how_many,
                                  (fftw_complex*) in_data, 0,
                                  in_stride, in_dist,
                                  (fftw_complex*) out_data, 0,
                                  out_stride, out_dist,
                                  FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(p);
  fftw_destroy_plan(p);
  fftw_cleanup_threads();
  
  return CPHVB_SUCCESS;
}


cphvb_error cphvb_fft(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_fft_type *m_arg = (cphvb_fft_type *) arg;
    cphvb_array *out = m_arg->operand[0];
    cphvb_array *in = m_arg->operand[1];
    
    if(in->ndim > 2)
        return CPHVB_ERROR;
    
    if(cphvb_data_malloc(out) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
        
    if(cphvb_data_malloc(in) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
        
    switch (in->type)
    {
    	case CPHVB_COMPLEX64:
	    	return do_fft_complex64(in, out);
    	case CPHVB_COMPLEX128:
	    	return do_fft_complex128(in, out);
    	default:
            return CPHVB_ERROR;
	}  
}

/* ONE DIMENSIONAL TRANSFORMATIONS END */

/* TWO DIMENSIONAL TRANSFORMATIONS START */

cphvb_error do_fft2_complex64(cphvb_array* in, cphvb_array* out){

  cphvb_complex64* in_data;
  cphvb_complex64* out_data;
  
  cphvb_data_get(in, (cphvb_data_ptr*) &in_data);
  cphvb_data_get(out, (cphvb_data_ptr*) &out_data);
  

  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());
  fftwf_plan p;
  p = fftwf_plan_dft_2d(in->shape[0], in->shape[1], (fftwf_complex*)in_data, (fftwf_complex*)out_data, FFTW_FORWARD, FFTW_ESTIMATE);

  fftwf_execute(p);
  fftwf_destroy_plan(p);
  fftwf_cleanup_threads();
  
  return CPHVB_SUCCESS;
}

cphvb_error do_fft2_complex128(cphvb_array* in, cphvb_array* out){

  cphvb_complex128* in_data;
  cphvb_complex128* out_data;
  
  cphvb_data_get(in, (cphvb_data_ptr*) &in_data);
  cphvb_data_get(out, (cphvb_data_ptr*) &out_data);
  

  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fftw_plan p;
  p = fftw_plan_dft_2d(in->shape[0], in->shape[1], (fftw_complex*)in_data, (fftw_complex*)out_data, FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(p);
  fftw_destroy_plan(p);
  fftw_cleanup_threads();
  
  return CPHVB_SUCCESS;
}

cphvb_error cphvb_fft2(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_fft_type *m_arg = (cphvb_fft_type *) arg;
    cphvb_array *out = m_arg->operand[0];
    cphvb_array *in = m_arg->operand[1];
    
    
    if(in->ndim != 2 || in->stride[1] != 1 || in->stride[0] != in->shape[1])
        return CPHVB_ERROR;
    
    if(cphvb_data_malloc(out) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
        
    if(cphvb_data_malloc(in) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
        
    switch (in->type)
    {
    	case CPHVB_COMPLEX64:
	    	return do_fft2_complex64(in, out);
    	case CPHVB_COMPLEX128:
	    	return do_fft2_complex128(in, out);
    	default:
            return CPHVB_ERROR;
	}  
}

/* ONE DIMENSIONAL TRANSFORMATIONS END */

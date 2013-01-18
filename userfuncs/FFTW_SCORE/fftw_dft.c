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
#include <fftw3.h>

/* ONE DIMENSIONAL TRANSFORMATIONS START */

bh_error do_fft_complex64(bh_array* in, bh_array* out){

  bh_complex64* in_data;
  bh_complex64* out_data;
  
  bh_data_get(in, (bh_data_ptr*) &in_data);
  bh_data_get(out, (bh_data_ptr*) &out_data);
  

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
    return BH_ERROR;
  fftwf_execute(p);
  fftwf_destroy_plan(p);
  
  return BH_SUCCESS;
}

bh_error do_fft_complex128(bh_array* in, bh_array* out){

  bh_complex128* in_data;
  bh_complex128* out_data;
  
  bh_data_get(in, (bh_data_ptr*) &in_data);
  bh_data_get(out, (bh_data_ptr*) &out_data);
  

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
  
  return BH_SUCCESS;
}


bh_error bh_fft(bh_userfunc *arg, void* ve_arg)
{
    bh_fft_type *m_arg = (bh_fft_type *) arg;
    bh_array *out = m_arg->operand[0];
    bh_array *in = m_arg->operand[1];
    
    if(in->ndim > 2)
        return BH_ERROR;
    
    if(bh_data_malloc(out) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
        
    if(bh_data_malloc(in) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
        
    switch (in->type)
    {
    	case BH_COMPLEX64:
	    	return do_fft_complex64(in, out);
    	case BH_COMPLEX128:
	    	return do_fft_complex128(in, out);
    	default:
            return BH_ERROR;
	}  
}

/* ONE DIMENSIONAL TRANSFORMATIONS END */

/* TWO DIMENSIONAL TRANSFORMATIONS START */

bh_error do_fft2_complex64(bh_array* in, bh_array* out){

  bh_complex64* in_data;
  bh_complex64* out_data;
  
  bh_data_get(in, (bh_data_ptr*) &in_data);
  bh_data_get(out, (bh_data_ptr*) &out_data);
  
  fftwf_plan p;
  p = fftwf_plan_dft_2d(in->shape[0], in->shape[1], (fftwf_complex*)in_data, (fftwf_complex*)out_data, FFTW_FORWARD, FFTW_ESTIMATE);

  fftwf_execute(p);
  fftwf_destroy_plan(p);
  
  return BH_SUCCESS;
}

bh_error do_fft2_complex128(bh_array* in, bh_array* out){

  bh_complex128* in_data;
  bh_complex128* out_data;
  
  bh_data_get(in, (bh_data_ptr*) &in_data);
  bh_data_get(out, (bh_data_ptr*) &out_data);
  
  fftw_plan p;
  p = fftw_plan_dft_2d(in->shape[0], in->shape[1], (fftw_complex*)in_data, (fftw_complex*)out_data, FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(p);
  fftw_destroy_plan(p);
  
  return BH_SUCCESS;
}

bh_error bh_fft2(bh_userfunc *arg, void* ve_arg)
{
    bh_fft_type *m_arg = (bh_fft_type *) arg;
    bh_array *out = m_arg->operand[0];
    bh_array *in = m_arg->operand[1];
    
    
    if(in->ndim != 2 || in->stride[1] != 1 || in->stride[0] != in->shape[1])
        return BH_ERROR;
    
    if(bh_data_malloc(out) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
        
    if(bh_data_malloc(in) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
        
    switch (in->type)
    {
    	case BH_COMPLEX64:
	    	return do_fft2_complex64(in, out);
    	case BH_COMPLEX128:
	    	return do_fft2_complex128(in, out);
    	default:
            return BH_ERROR;
	}  
}

/* TWO DIMENSIONAL TRANSFORMATIONS END */

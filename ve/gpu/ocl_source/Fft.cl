/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#define PI 3.14159265

float2 cmul(float2 c1, float2 c2){
   float2 ret;
   ret.x = c1.x * c2.x - c1.y * c2.y;
   ret.y = c1.x * c2.y + c1.y * c2.x;
   return ret;
}
float2 polar_to_rect(float r, float angle){
  float2 ret;
  ret.x = r * cos(angle);
  ret.y = r * sin(angle);
  return ret;
}
void dft2(float2* x1, float2* x2){
  float2 temp = *x1;
  *x1 = *x1 + *x2;
  *x2 = temp - *x2;
}

__kernel void fft(__global const float2* x, __global float2* out, long p){
  int i = get_global_id(0);
  int t = get_global_size(0);
  int k = i & (p-1);
  
  float2 c0 = x[i];
  float2 c1 = x[i+t];
  
  float2 twiddle = polar_to_rect(1, -1 * PI * k / p);
  c1 = cmul(c1, twiddle);
  
  dft2(&c0, &c1);
  
  int index = (i << 1) - k;
  out[index] = c0;
  out[index+p] = c1;
}

__kernel void fft2d(__global const float2* x, __global float2* out, long p, long s0, long s1){
  int i = get_global_id(0);
  int j = get_global_id(1);
  int t = get_global_size(0);
  int k = i & (p-1);
  
  float2 c0 = x[i*s0 + j*s1];
  float2 c1 = x[(i+t)*s0 + j*s1];
  
  float2 twiddle = polar_to_rect(1, -1 * PI * k / p);
  c1 = cmul(c1, twiddle);
  
  dft2(&c0, &c1);
  
  int index = (i << 1) - k;
  out[index * s0 + j*s1] = c0;
  out[(index+p)*s0 + j*s1] = c1;
}

__kernel void copy(__global const float2* in, __global float2* out){
  int i = get_global_id(0);
  out[i] = in[i];  
}

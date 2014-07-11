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

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/*
  Implementing matrix multiplication:
  C(ds1xds0) = A(ds1xds2)*B(ds2xds0)
 */

#define MATMUL_TMPL(dtype,mulexpr)                  \
__kernel void matmul_##dtype(                       \
              const int ds0                         \
            , const int ds1                         \
            , const int ds2                         \
            , const int v0s2                        \
            , const int v0s1                        \
            , const int v0s0                        \
            , const int v1s2                        \
            , const int v1s1                        \
            , const int v1s0                        \
            , const int v2s2                        \
            , const int v2s1                        \
            , const int v2s0                        \
            , __global dtype *C                     \
            , __global dtype *A                     \
            , __global dtype *B )                   \
{                                                   \
    const size_t gidx = get_global_id(0);           \
	if (gidx >= ds0)                                \
		return;                                     \
    const size_t gidy = get_global_id(1);           \
	if (gidy >= ds1)                                \
		return;                                     \
    dtype c = 0;                                    \
    for (int ids2 = 0; ids2 < ds2; ++ids2)          \
    {                                               \
        dtype a = A[gidy*v1s2 + ids2*v1s1 + v1s0];  \
        dtype b = B[ids2*v2s2 + gidx*v2s1 + v2s0];  \
        c += mulexpr;                               \
    }                                               \
    C[gidy*v0s2 + gidx*v0s1 + v0s0] = c;            \
}

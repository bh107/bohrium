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

inline uint TausStep(uint z, int s0, int s1, int s2, uint M)
{
    uint b = (((z << s0) ^ z) >> s1);
    return (((z & M) << s2) ^ b);  
}

inline uint LCGStep(uint z, uint A, uint C)
{
    return (A * z + C);    
}

inline uint HybridTaus(uint4* z)
{
    (*z).s0 = TausStep((*z).s0, 13, 19, 12, 4294967294U);
    (*z).s1 = TausStep((*z).s1, 2, 25, 4, 4294967288U);
    (*z).s2 = TausStep((*z).s2, 3, 11, 17, 4294967280U);
    (*z).s3 = LCGStep((*z).s3, 1664525, 1013904223U);
    return ((*z).s0 ^ (*z).s1 ^ (*z).s2 ^ (*z).s3);
}

__kernel void htrand_uint32(__global uint* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        res[i] = HybridTaus(&z);
    }
    state[gidx] = z;
}

__kernel void htrand_int32(__global int* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        res[i] = HybridTaus(&z) >> 1;
    }
    state[gidx] = z;
}

__kernel void htrand_float32(__global float* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        res[i] = (float)HybridTaus(&z) * 2.3283064365387e-10;
    }
    state[gidx] = z;
}

__kernel void htrand_uint64(__global ulong* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        ulong r = HybridTaus(&z);
        res[i] = (r << 32) | HybridTaus(&z);
    }
    state[gidx] = z;
}

__kernel void htrand_int64(__global long* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        long r = (HybridTaus(&z) >> 1);
        res[i] = (r << 32) | HybridTaus(&z);
    }
    state[gidx] = z;
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void htrand_float64(__global double* res, long size, __global uint4* state)
{
    
    const size_t gsize = get_global_size(0);
    uint gidx = get_global_id(0);
    uint4 z = state[gidx];
    for (size_t i = gidx; i < size; i += gsize) 
    {
        /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
        long r1 = (HybridTaus(&z) >> 5);
        long r2 = (HybridTaus(&z) >> 6);
        res[i] = (r1 * 67108864.0 + r2) / 9007199254740992.0;
    }
    state[gidx] = z;
}
#endif

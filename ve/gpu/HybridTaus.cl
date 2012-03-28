/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

inline uint TausStep(uint* z, int s0, int s1, int s2, uint M)
{
    uint b = (((*z << s0) ^ *z) >> s1);
    return *z = (((*z & M) << s2) ^ b);  
}

inline uint LCGStep(uint* z, uint A, uint C)
{
    return *z = (A * *z + C);    
}

inline uint HybridTaus(uint4* z)
{
    return TausStep(&(z->s0), 13, 19, 12, 4294967294U) ^ 
        TausStep(&(z->s1), 2, 25, 4, 4294967288U)      ^
        TausStep(&(z->s2), 3, 11, 17, 4294967280U)     ^
        LCGStep(&(z->s3), 1664525, 1013904223U);
}

__kernel void htrand_uint32(uint* res, long size, uint4* state)
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

__kernel void htrand_int32(int* res, long size, uint4* state)
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

__kernel void htrand_float32(float* res, long size, uint4* state)
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

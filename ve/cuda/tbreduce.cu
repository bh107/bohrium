/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

extern "C" __global__ 
void add_reduce_float_1d(float* in, uint elements, float* out)
{
    const uint nThreads = blockDim.x*gridDim.x;
	const uint outIdx = threadIdx.x + blockIdx.x*blockDim.x;
    float myRes = 0.0;
    uint inIdx;
	for (inIdx = outIdx; inIdx < elements; inIdx += nThreads) 
	{
        myRes += in[inIdx];
	}
    out[outIdx] = myRes;
}



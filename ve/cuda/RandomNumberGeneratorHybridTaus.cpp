/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

#include <cstdlib>
#include <ctime>
#include <cassert>
#include "RandomNumberGeneratorHybridTaus.hpp"
#include "KernelPredefined.hpp"
#include "WrapperFunctions.hpp"

void RandomNumberGeneratorHybridTaus::initState()
{
    //Generate the initial state (Random numbers)
    srandom((unsigned int)std::time(NULL));
    for (int i = 0; i < HT_TPB*HT_BPG; ++i)
    {
        for (int z = 0; z < 4; ++z)
        {
            state[i][z] = random();
            while (state[i][z] < 129)
            {
                state[i][z] = random();
            }
        }
    }
    
    //upload the state to the cuda device
    size_t size = HT_TPB * HT_BPG * 4 * sizeof(uint);
    CUresult error = cuMemAlloc(&cudaState, size);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not allocate memory for random state on device");
    }
    error = cuMemcpyHtoD(cudaState, &state, size);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not copy random state to device.");
    }
}

RandomNumberGeneratorHybridTaus::RandomNumberGeneratorHybridTaus() :
    shape(new KernelShape(HT_TPB,1,1,HT_BPG,1))
{
    initState();
    CUmodule module = KernelPredefined::loadSource("/usr/local/cphvb/sourcekernels/htrand.ptx");
    htrand_float32 = new KernelPredefined(module, "htrand_float32", 
                                        {PTX_POINTER, PTX_UINT32, PTX_POINTER});
    htrand_uint32 = new KernelPredefined(module, "htrand_uint32",  
                                        {PTX_POINTER, PTX_UINT32, PTX_POINTER});
    htrand_int32 = new KernelPredefined(module, "htrand_int32",  
                                        {PTX_POINTER, PTX_UINT32, PTX_POINTER});
    htrand_float32_step = new KernelPredefined(module, "htrand_float32_step",  
                            {PTX_POINTER, PTX_UINT32, PTX_UINT32, PTX_POINTER});
    htrand_uint32_step = new KernelPredefined(module, "htrand_uint32_step",   
                            {PTX_POINTER, PTX_UINT32, PTX_UINT32, PTX_POINTER});
    htrand_int32_step = new KernelPredefined(module, "htrand_int32_step",   
                            {PTX_POINTER, PTX_UINT32, PTX_UINT32, PTX_POINTER});

}

RandomNumberGeneratorHybridTaus::~RandomNumberGeneratorHybridTaus()
{
    CUresult error = cuMemFree(cudaState);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not free device memory.");
    }
}

void RandomNumberGeneratorHybridTaus::fill(cphVBarray* array)
{
    ParameterList parameters;
    cphVBarray* base = cphVBBaseArray(array);
    int elements = cphvb_nelements(base->ndim, base->shape);
    parameters.push_back(KernelParameter(PTX_POINTER, base->cudaPtr));
    parameters.push_back(KernelParameter(PTX_UINT32, elements));
    parameters.push_back(KernelParameter(PTX_POINTER, cudaState));
    switch (base->cudaType)
    {
    case PTX_FLOAT32:
        htrand_float32->execute(parameters,shape);
        break;
    case PTX_INT32:
        htrand_int32->execute(parameters,shape);
        break;
    case PTX_UINT32:
        htrand_uint32->execute(parameters,shape);
        break;
    default:
        assert(false);
    }
}

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

#include <cassert>
#include "ReduceTB.hpp"
#include "KernelPredefined.hpp"
#include "WrapperFunctions.hpp"

ReduceTB::ReduceTB() :
    shape(new KernelShape(R_TPB,1,1,R_BPG,1))
{
    CUmodule module = KernelPredefined::loadSource("/usr/local/cphvb/tbreduce.ptx");
    add_reduce_float_1d = new KernelPredefined(module, "add_reduce_float_1d", 
                                     {PTX_POINTER, PTX_UINT32, PTX_POINTER});
    //reserve space for intermediate result
    size_t size = R_TPB * R_BPG * sizeof(float);
    CUresult error = cuMemAlloc(&cudaRes, size);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not allocate memory for intermediate result on device");
    }
}

ReduceTB::~ReduceTB()
{
    CUresult error = cuMemFree(cudaRes);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not free device memory.");
    }
}

void ReduceTB::reduce(cphVBarray* resArray, cphVBarray* array)
{
    ParameterList parameters;
    cphVBarray* base = cphVBBaseArray(array);
    int elements = cphvb_nelements(base->ndim, base->shape);
    parameters.push_back(KernelParameter(PTX_POINTER, base->cudaPtr));
    parameters.push_back(KernelParameter(PTX_UINT32, elements));
    parameters.push_back(KernelParameter(PTX_POINTER, cudaRes));
    switch (base->cudaType)
    {
    case PTX_FLOAT32:
        add_reduce_float_1d->execute(parameters,shape);
        break;
    default:
        assert(false);
    }
    //Copy intermediate result to host
    size_t size = R_TPB * R_BPG * sizeof(float);
    CUresult error = cuMemcpyDtoH(&hostRes, cudaRes, size);
    if (error !=  CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not copy intermediate result to host.");
    }
    //Do final recuction on host
    float res = 0.0;
    for (int i = 0; i < R_TPB * R_BPG; ++i)
    {
        res += hostRes[i];
    }
    resArray->init_value.float32 = res;
    resArray->data = &resArray->init_value;
}

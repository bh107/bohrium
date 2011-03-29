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

#include "CUDAerrorCode.h"
#include "KernelPredefined.hpp"

KernelPredefined::KernelPredefined(CUmodule module_,
                                   const char* functionName,
                                   Signature signature_)
{
    module = module_;
    CUresult error = cuModuleGetFunction(&entry, module, functionName);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not get function name.");
    }
    signature = signature_; 
}

CUmodule KernelPredefined::loadSource(const char* fileName)
{
    CUmodule module;
    CUresult error = cuModuleLoad(&module, fileName);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not load module from source.");
    }
    return module;
}

void KernelPredefined::execute(ParameterList parameters, KernelShape* shape)
{
    setParameters(parameters);
    launchGrid(shape);
}

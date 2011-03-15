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

#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <stdexcept>
#include "PTXtype.h"
#include "KernelParameter.hpp"
#include "Kernel.hpp"
#include "KernelShape.hpp"
#include "CUDAerrorCode.h"


#define ALIGN_UP(offset, alignment) \
	(offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

Kernel::Kernel() {}

Kernel::Kernel(CUmodule module_,
               CUfunction entry_,
               Signature signature_) :
    module(module_),
    entry(entry_),
    signature(signature_) {}

#define JIT_LOG_BUFFER_SIZE (4096)
char jitInfoLogBuffer[JIT_LOG_BUFFER_SIZE];
char jitErrorLogBuffer[JIT_LOG_BUFFER_SIZE];

Kernel::Kernel(PTXkernel* ptxKernel)
{
    std::ostringstream ptxKernelSource;
    ptxKernelSource << *ptxKernel;
    std::string ks = ptxKernelSource.str();
    const char* cks = ks.c_str();

    const unsigned int options = 4;
    CUjit_option jitOptions[options];
    void* jitValues[options];

	jitOptions[0] = CU_JIT_INFO_LOG_BUFFER;
	jitValues[0] = jitInfoLogBuffer;
	jitOptions[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	jitValues[1] = (void *)JIT_LOG_BUFFER_SIZE;
	jitOptions[2] = CU_JIT_ERROR_LOG_BUFFER;
	jitValues[2] = jitErrorLogBuffer;
	jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	jitValues[3] = (void *)JIT_LOG_BUFFER_SIZE;

    CUresult error = cuModuleLoadDataEx(&module, cks, 
                                        options, jitOptions, jitValues);      
#ifdef DEBUG
    std::cout << "[VE CUDA] Compilation info:" << std::endl;
    std::cout << "--- INFO BEGIN ---" << std::endl;
    std::cout << jitInfoLogBuffer << std::endl;
    std::cout << "---- INFO END ----" << std::endl;		
    std::cout << "--- ERROR BEGIN ---" << std::endl;
    std::cout << jitErrorLogBuffer << std::endl;
    std::cout << "---- ERRORS END ----" << std::endl;		
    std::cout << "---- CODE BEGIN ----" << std::endl;
    std::cout << cks << std::endl;
    std::cout << "----- CODE END -----" << std::endl;	
#endif

    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not compile kernel");
    }
    
    error = cuModuleGetFunction(&entry, module, ptxKernel->name);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not get function name.");
    }
    signature = ptxKernel->getSignature();
}


void Kernel::setParameters(ParameterList parameters)
{
    CUresult error;
    int offset = 0;
    assert (parameters.size() == signature.size());
    Signature::iterator siter = signature.begin();
    ParameterList::iterator piter = parameters.begin();
    while (piter != parameters.end())
    {
        assert (piter->type == *siter);
        ALIGN_UP(offset, ptxAlign(piter->type));
        error = cuParamSetv(entry, offset, &piter->value, 
                            ptxSizeOf(piter->type));
        if (error !=  CUDA_SUCCESS)
        {
#ifdef DEBUG
            std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
            throw std::runtime_error("Could not set kernel parameter.");
        }
        offset += ptxSizeOf(piter->type);
        ++piter; ++siter;
    }
    error = cuParamSetSize(entry,offset);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not set kernel parameter list size.");
    }
}

void Kernel::setBlockShape(int x, int y, int z)
{
    CUresult error = cuFuncSetBlockShape(entry, x, y, y);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
#endif
        throw std::runtime_error("Could not set kernel block shape.");
    }
}

void Kernel::launchGrid(KernelShape* shape)
{
    setBlockShape(shape->threadsPerBlockX, 
                  shape->threadsPerBlockY, shape->threadsPerBlockY);
    CUresult error = cuLaunchGrid(entry, shape->blocksPerGridX, 
                                         shape->blocksPerGridY);
    if (error !=  CUDA_SUCCESS)
    {
#ifdef DEBUG
        std::cout << "[VE CUDA] " << cudaErrorStr(error) << std::endl;
        std::cout << "KernelShape : " << std::endl <<
            "\tthreadsPerBlockX: " << shape->threadsPerBlockX << std::endl << 
            "\tthreadsPerBlockY: " << shape->threadsPerBlockY << std::endl << 
            "\tthreadsPerBlockZ: " << shape->threadsPerBlockZ << std::endl << 
            "\tblocksPerGridX: " << shape->blocksPerGridX << std::endl << 
            "\tblocksPerGridY: " << shape->blocksPerGridY << std::endl; 
#endif
        throw std::runtime_error("Could not set kernel grid.");
    }
}

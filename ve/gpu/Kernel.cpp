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

#include <cassert>
#include <stdexcept>
#include "Kernel.hpp"

Kernel::Kernel(ResourceManager* resourceManager_, 
               cphvb_intp ndim_,
               const std::vector<OCLtype>& signature_,
               const std::string& source, 
               const std::string& name)
    : resourceManager(resourceManager_)
    , ndim(ndim_)
    , signature(signature_)
{
    kernel = resourceManager->createKernel(source.c_str(), name.c_str());
}

void Kernel::call(ArrayArgs& arrayArgs,
                  const std::vector<Scalar>& scalarArgs,
                  const std::vector<cphvb_index>& shape)
{
    unsigned int device = 0;
    cl::NDRange globalSize;
    switch (shape.size())
    {
    case 1:
        globalSize = cl::NDRange(shape[0]);
        break;
    case 2:    
        globalSize = cl::NDRange(shape[0], shape[1]);
        break;
    case 3:    
        globalSize = cl::NDRange(shape[0], shape[1], shape[2]);
        break;
    default:
        throw std::runtime_error("More than 3 dimensions not supported.");
    }
    std::vector<cl::Event> waitFor;
    for (ArrayArgs::iterator aait = arrayArgs.begin(); aait != arrayArgs.end(); ++aait)
    {
        waitFor.push_back(aait->first->getWriteEvent());
    }
    unsigned int argIndex = 0;
    for (ArrayArgs::iterator aait = arrayArgs.begin(); aait != arrayArgs.end(); ++aait)
        kernel.setArg(argIndex++, aait->first->getBuffer());
    for (std::vector<Scalar>::const_iterator sait = scalarArgs.begin(); sait != scalarArgs.end(); ++sait)
        sait->addToKernel(kernel, argIndex++);
    cl::Event event = resourceManager->enqueueNDRangeKernel(kernel, globalSize, &waitFor ,device); 
    for (ArrayArgs::iterator aait = arrayArgs.begin(); aait != arrayArgs.end(); ++aait)
    {
        if (aait->second)
            aait->first->setWriteEvent(event);
    }
}

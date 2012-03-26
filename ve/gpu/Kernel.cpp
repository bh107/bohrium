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
               const std::string& source, 
               const std::string& name)
    : resourceManager(resourceManager_)
    , ndim(ndim_)
{
    kernel = resourceManager->createKernel(source.c_str(), name.c_str());
}

void Kernel::call(Parameters& parameters,
                  const std::vector<cphvb_index>& globalShape)
{
    call(parameters, globalShape, resourceManager->localShape(globalShape.size()));
}

void Kernel::call(Parameters& parameters,
                  const std::vector<cphvb_index>& globalShape,
                  const std::vector<size_t>& localShape)
{
    assert(globalShape.size() == localShape.size());
    unsigned int device = 0;
    cl::NDRange globalSize, localSize;
    int rem0, rem1, rem2;
    switch (globalShape.size())
    {
    case 1:
        localSize = cl::NDRange(localShape[0]);
        rem0 = globalShape[0] % localShape[0];
        globalSize = cl::NDRange(globalShape[0] + (rem0==0?0:(localShape[0]-rem0)));
        break;
    case 2:    
        localSize = cl::NDRange(localShape[0], localShape[1]);
        rem0 = globalShape[0] % localShape[0];
        rem1 = globalShape[1] % localShape[1];
        globalSize = cl::NDRange(globalShape[0] + (rem0==0?0:(localShape[0]-rem0)),
                                 globalShape[1] + (rem1==0?0:(localShape[1]-rem1)));
        break;
    case 3:    
        localSize = cl::NDRange(localShape[0], localShape[1], localShape[2]);
        rem0 = globalShape[0] % localShape[0];
        rem1 = globalShape[1] % localShape[1];
        rem2 = globalShape[2] % localShape[2];
        globalSize = cl::NDRange(globalShape[0] + (rem0==0?0:(localShape[0]-rem0)),
                                 globalShape[1] + (rem1==0?0:(localShape[1]-rem1)),
                                 globalShape[2] + (rem2==0?0:(localShape[2]-rem2)));
        break;
    default:
        throw std::runtime_error("More than 3 dimensions not supported.");
    }
    std::vector<cl::Event> waitFor;
    for (Parameters::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
    {
        if (!pit->first->isScalar())
        {// If its not a scalar wait for any write events 
            waitFor.push_back(pit->first->getWriteEvent());
            if (pit->second)
            {// If we are going to write to it: Also wait for any read events
                std::deque<cl::Event> re = pit->first->getReadEvents();
                for (std::deque<cl::Event>::iterator reit = re.begin(); reit != re.end(); ++reit)
                    waitFor.push_back(*reit);
            }
        }
    }
    unsigned int argIndex = 0;
    for (Parameters::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
        pit->first->addToKernel(true, kernel, argIndex++);
    cl::Event event = resourceManager->enqueueNDRangeKernel(kernel, globalSize, localSize, &waitFor ,device); 
    for (Parameters::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
    {
        if (pit->second)
            pit->first->setWriteEvent(event);
        else
            pit->first->addReadEvent(event);
    }
}

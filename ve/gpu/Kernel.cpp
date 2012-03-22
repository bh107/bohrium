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
                  const std::vector<cphvb_index>& shape)
{
    unsigned int device = 0;
    cl::NDRange globalSize, localSize;
    switch (shape.size())
    {
    case 1:
        localSize = cl::NDRange(256);
        globalSize = cl::NDRange(shape[0] + ((shape[0] % 256)==0?0:256-(shape[0] % 256)));
        break;
    case 2:    
        localSize = cl::NDRange(32,16);
        globalSize = cl::NDRange(shape[0] + ((shape[0] % 32)==0?0:32-(shape[0] % 32)), 
                                 shape[1] + ((shape[1] % 16)==0?0:16-(shape[1] % 16)));
        break;
    case 3:    
        localSize = cl::NDRange(32,4,4);
        globalSize = cl::NDRange(shape[0] + ((shape[0] % 32)==0?0:32-(shape[0] % 32)), 
                                 shape[1] + ((shape[1] % 4)==0?0:4-(shape[1] % 4)),
                                 shape[2] + ((shape[2] % 4)==0?0:4-(shape[2] % 4)));
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

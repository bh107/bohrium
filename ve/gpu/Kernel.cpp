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

#include <cassert>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include "Kernel.hpp"

Kernel::Kernel(ResourceManager* resourceManager_, 
               cphvb_intp ndim_,
               const std::string& source, 
               const std::string& name)
    : resourceManager(resourceManager_)
    , ndim(ndim_)
{
    assert(ndim > 0);
    kernel = resourceManager->createKernel(source , name);
}

Kernel::Kernel(ResourceManager* resourceManager_, cphvb_intp ndim_, cl::Kernel kernel_)
    : resourceManager(resourceManager_)
    , ndim(ndim_)
    , kernel(kernel_) {}



std::vector<Kernel> Kernel::createKernels(ResourceManager* resourceManager, 
                                          const std::vector<cphvb_intp> ndims,
                                          const std::string& source, 
                                          const std::vector<std::string>& kernelNames)
{
    assert(ndims.size() == kernelNames.size());
    std::vector<cl::Kernel> clKernels = resourceManager->createKernels(source, kernelNames);
    std::vector<Kernel> kernels;
    for (size_t i = 0; i < clKernels.size(); ++i)
    {
        kernels.push_back(Kernel(resourceManager, ndims[i], clKernels[i]));
    }
    return kernels;
}
 
std::vector<Kernel> Kernel::createKernelsFromFile(ResourceManager* resourceManager, 
                                                  const std::vector<cphvb_intp> ndims,
                                                  const std::string& fileName, 
                                                  const std::vector<std::string>& kernelNames)
{
    std::ifstream file(fileName.c_str(), std::ios::in);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open source file.");
    }
    std::ostringstream source;
    source << file.rdbuf();
    return createKernels(resourceManager, ndims, source.str(), kernelNames);
}


void Kernel::call(Parameters& parameters,
                  const std::vector<size_t>& globalShape)
{
    call(parameters, globalShape, resourceManager->localShape(globalShape));
}

void Kernel::call(Parameters& parameters,
                  const std::vector<size_t>& globalShape,
                  const std::vector<size_t>& localShape)
{
    assert(globalShape.size() ==  (size_t)ndim && localShape.size() == (size_t)ndim);
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
        Buffer* ba = dynamic_cast<Buffer*>(pit->first);
        if (ba)
        {// If its not a scalar wait for any write events 
            waitFor.push_back(ba->getWriteEvent());
            if (pit->second)
            {// If we are going to write to it: Also wait for any read events
                std::deque<cl::Event> re = ba->getReadEvents();
                for (std::deque<cl::Event>::iterator reit = re.begin(); reit != re.end(); ++reit)
                    waitFor.push_back(*reit);
            }
        }
    }
    unsigned int argIndex = 0;
    for (Parameters::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
        pit->first->addToKernel(kernel, argIndex++);
    cl::Event event = resourceManager->enqueueNDRangeKernel(kernel, globalSize, localSize, &waitFor ,device); 
    for (Parameters::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
    {
        Buffer* ba = dynamic_cast<Buffer*>(pit->first);
        if (ba)
        {
            if (pit->second)
                ba->setWriteEvent(event);
            else 
                ba->addReadEvent(event);

        }
    }
}

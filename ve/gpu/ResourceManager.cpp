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

#include "ResourceManager.hpp"
#include <stdexcept>

ResourceManager::ResourceManager()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    bool foundPlatform = false;
    for (cl::Platform platform: platforms)
    {
        try {
            cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),0};
            context = cl::Context(CL_DEVICE_TYPE_GPU, props);
            foundPlatform = true;
            break;
        } 
        catch (cl::Error)
        {
            foundPlatform = false;
        }
    }
    if (foundPlatform)
    {
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        for(cl::Device& device: devices)        
        {
            commandQueues.push_back(cl::CommandQueue(context,device,0));
        }
    } else {
        throw std::runtime_error("Could not find valid OpenCL platform.");
    }
}

cl::Buffer ResourceManager::createBuffer(size_t size)
{
    return cl::Buffer(context, CL_MEM_READ_WRITE, size, NULL);
}

cl::Event ResourceManager::enqueueReadBuffer(const cl::Buffer buffer,
                                             void* hostPtr, 
                                             const std::vector<cl::Event>* waitFor,
                                             int device)
{
    cl::Event event;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    commandQueues[device].enqueueReadBuffer(buffer, CL_TRUE, 0, size, hostPtr, waitFor, &event);
    return event;
}

cl::Event ResourceManager::enqueueWriteBuffer(const cl::Buffer buffer,
                                              const void* hostPtr, 
                                              const std::vector<cl::Event>* waitFor,
                                              int device)
{
    cl::Event event;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    commandQueues[device].enqueueWriteBuffer(buffer, CL_TRUE, 0, size, hostPtr, waitFor, &event);
    return event;
}

cl::Kernel ResourceManager::createKernel(const char* source, const char* kernelName)
{
    cl::Program program(context,cl::Program::Sources(1,std::make_pair(source,0)));
    program.build(devices);
    return cl::Kernel(program, kernelName);
}

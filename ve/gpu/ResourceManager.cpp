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
#include <cassert>
#include <stdexcept>
#include <iostream>

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
        maxWorkGroupSize = 1 << 16;
        for(cl::Device& device: devices)        
        {
            commandQueues.push_back(cl::CommandQueue(context,device,
                                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#ifdef STATS
                                                     || CL_QUEUE_PROFILING_ENABLE
#endif
                                        ));
            size_t mwgs = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            maxWorkGroupSize = maxWorkGroupSize>mwgs?mwgs:maxWorkGroupSize; 
        }
    } else {
        throw std::runtime_error("Could not find valid OpenCL platform.");
    }
    
#ifdef STATS
    batchBuild = 0.0;
    batchSource = 0.0;
    resourceCreateKernel = 0.0;
#endif
}

#ifdef STATS
ResourceManager::~ResourceManager()
{
    std::cout << "------------------ STATS ------------------------" << std::endl;
    std::cout << "Batch building:           " << batchBuild / 1000000 << std::endl;
    std::cout << "Source generation:        " << batchSource / 1000000 << std::endl;
    std::cout << "OpenCL kernel generation: " << resourceCreateKernel / 1000000 << std::endl;
}
#endif

cl::Buffer ResourceManager::createBuffer(size_t size)
{
    return cl::Buffer(context, CL_MEM_READ_WRITE, size, NULL);
}

void ResourceManager::readBuffer(const cl::Buffer& buffer,
                                 void* hostPtr, 
                                 cl::Event waitFor,
                                 unsigned int device)
{
    std::cout << "readBuffer()" << std::endl;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    std::vector<cl::Event> readerWaitFor;
    readerWaitFor.push_back(waitFor);
    try {
        commandQueues[device].enqueueReadBuffer(buffer, CL_TRUE, 0, size, hostPtr, &readerWaitFor, NULL);
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueReadBuffer: \"" << e.err() << "\"" << std::endl;
    }
}

cl::Event ResourceManager::enqueueWriteBuffer(const cl::Buffer& buffer,
                                              const void* hostPtr, 
                                              unsigned int device)
{
    std::cout << "enqueueWriteBuffer()" << std::endl;
    cl::Event event;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    try {
        commandQueues[device].enqueueWriteBuffer(buffer, CL_FALSE, 0, size, hostPtr, NULL, &event);
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueWriteBuffer: \"" << e.what() << "\"" << std::endl;
    }
    return event;
}

cl::Event ResourceManager::completeEvent()
{
    cl::UserEvent event(context);
    event.setStatus(CL_COMPLETE);
    return event;
}

cl::Kernel ResourceManager::createKernel(const char* source, const char* kernelName)
{
#ifdef STATS
    timeval start, end;
    gettimeofday(&start,NULL);
#endif
#ifdef DEBUG
    std::cout << "Kernel build :\n";
    std::cout << "------------------- SOURCE -----------------------\n";
    std::cout << source;
    std::cout << "------------------ SOURCE END --------------------" << std::endl;
#endif
    cl::Program::Sources sources(1,std::make_pair(source,0));
    cl::Program program(context, sources);
    try {
        program.build(devices);
    } catch (cl::Error) {
#ifdef DEBUG
        std::cerr << "Kernel build error:\n";
        std::cerr << "------------------- SOURCE -----------------------\n";
        std::cerr << source;
        std::cerr << "------------------ SOURCE END --------------------\n";
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
#endif
        throw std::runtime_error("Could not build Kernel.");
    }
    
    cl::Kernel kernel(program, kernelName);
#ifdef STATS
    gettimeofday(&end,NULL);
    resourceCreateKernel += (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_usec - start.tv_usec);
#endif
    return kernel;
}

cl::Event ResourceManager::enqueueNDRangeKernel(const cl::Kernel& kernel, 
                                                const cl::NDRange& globalSize,
                                                const cl::NDRange& localSize,
                                                const std::vector<cl::Event>* waitFor,
                                                unsigned int device)
{
    cl::Event event;
    commandQueues[device].enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, waitFor, &event);
    return event;
}

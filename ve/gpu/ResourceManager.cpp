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
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#define STD_MIN(a, b) ((a) < (b) ? (a) : (b))
#define STD_MAX(a, b) ((a) >= (b) ? (a) : (b))
#else
#define STD_MIN(a, b) std::min(a, b)
#define STD_MAX(a, b) std::max(a, b)
#endif

ResourceManager::ResourceManager(cphvb_component* _component) 
    : component(_component)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    bool foundPlatform = false;
    for(std::vector<cl::Platform>::iterator pit = platforms.begin(); pit != platforms.end(); ++pit)        
    {
        try {
            cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(*pit)(),0};
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
        for(std::vector<cl::Device>::iterator dit = devices.begin(); dit != devices.end(); ++dit)        
        {
            commandQueues.push_back(cl::CommandQueue(context,*dit,
                                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#ifdef STATS
                                                     | CL_QUEUE_PROFILING_ENABLE
#endif
                                        ));
            if (dit == devices.begin())
            {
                maxWorkGroupSize = dit->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                maxWorkItemDims = dit->getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
                maxWorkItemSizes = dit->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES >();
            }
            else {
                size_t mwgs = dit->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                maxWorkGroupSize = STD_MIN(maxWorkGroupSize,mwgs);
                cl_uint mwid = dit->getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
                maxWorkItemDims = STD_MIN(maxWorkItemDims,mwid);
                std::vector<size_t> mwis = dit->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES >();
                for (cl_uint d = 0; d < maxWorkItemDims; ++d)
                    maxWorkItemSizes[d] = STD_MIN(maxWorkItemSizes[d],mwis[d]);
            }
        }
    } else {
        throw std::runtime_error("Could not find valid OpenCL platform.");
    }
    
    // Calculate "sane" localShapes
    size_t lsx = STD_MIN(256UL,maxWorkItemSizes[0]);
#ifdef DEBUG
    std::cout << "ResourceManager.localShape1D[" << lsx << "]" << std::endl;
#endif
    localShape1D.push_back(lsx);
    lsx = STD_MIN(32UL,maxWorkItemSizes[0]);
    size_t lsy = STD_MIN(maxWorkGroupSize/lsx,maxWorkItemSizes[1]);
#ifdef DEBUG
    std::cout << "ResourceManager.localShape2D[" << lsx << ", " << lsy << "]" << std::endl;
#endif
    localShape2D.push_back(lsx);
    localShape2D.push_back(lsy);
    lsx = STD_MIN(16UL,maxWorkItemSizes[0]);
    lsy = 1;
    while(lsy < std::sqrt((float)(maxWorkGroupSize/lsx)))
        lsy <<= 1;
    lsy = STD_MIN(lsy,maxWorkItemSizes[1]);
    size_t lsz = STD_MIN(maxWorkGroupSize/(lsx*lsy),maxWorkItemSizes[2]); 
#ifdef DEBUG
    std::cout << "ResourceManager.localShape3D[" << lsx << ", " << lsy << ", " << lsz << "]" << std::endl;
#endif
    localShape3D.push_back(lsx);
    localShape3D.push_back(lsy);
    localShape3D.push_back(lsz);


#ifdef STATS
    batchBuild = 0.0;
    batchSource = 0.0;
    resourceCreateKernel = 0.0;
    resourceBufferWrite = 0.0;
    resourceBufferRead = 0.0;
    resourceKernelExecute = 0.0;
#endif
}

#ifdef STATS
ResourceManager::~ResourceManager()
{
    std::cout << std::fixed;
    std::cout << "------------------ STATS ------------------------" << std::endl;
    std::cout << "Batch building:           " << batchBuild / 1000000.0 << std::endl;
    std::cout << "Source generation:        " << batchSource / 1000000.0 << std::endl;
    std::cout << "OpenCL kernel generation: " << resourceCreateKernel / 1000000.0 << std::endl;
    std::cout << "Writing buffers:          " << resourceBufferWrite / 1000000.0 << std::endl;
    std::cout << "Reading buffers:          " << resourceBufferRead / 1000000.0 << std::endl;
    std::cout << "Executing kernels:        " << resourceKernelExecute / 1000000.0 << std::endl;
}
#endif

cl::Buffer ResourceManager::createBuffer(size_t size)
{
    return cl::Buffer(context, CL_MEM_READ_WRITE, size);
}

void ResourceManager::readBuffer(const cl::Buffer& buffer,
                                 void* hostPtr, 
                                 cl::Event waitFor,
                                 unsigned int device)
{
#ifdef DEBUG
    std::cout << "readBuffer(" << hostPtr << ")" << std::endl;
#endif
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    std::vector<cl::Event> readerWaitFor(1,waitFor);
#ifdef STATS
    cl::Event event;
#endif
    try {
        commandQueues[device].enqueueReadBuffer(buffer, CL_TRUE, 0, size, hostPtr, &readerWaitFor, 
#ifdef STATS
                                                &event
#else
                                                NULL
#endif
            );
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueReadBuffer: \"" << e.err() << "\"" << std::endl;
    }
#ifdef STATS
    event.setCallback(CL_COMPLETE, &eventProfiler, &resourceBufferRead);
#endif
}

cl::Event ResourceManager::enqueueWriteBuffer(const cl::Buffer& buffer,
                                              const void* hostPtr, 
                                              std::vector<cl::Event> waitFor, 
                                              unsigned int device)
{
#ifdef DEBUG
    std::cout << "enqueueWriteBuffer(" << hostPtr << ")" << std::endl;
#endif
    cl::Event event;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    try {
        commandQueues[device].enqueueWriteBuffer(buffer, CL_FALSE, 0, size, hostPtr, &waitFor, &event);
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueWriteBuffer: \"" << e.what() << "\"" << std::endl;
        throw e;
    }
#ifdef STATS
    event.setCallback(CL_COMPLETE, &eventProfiler, &resourceBufferWrite);
#endif
    return event;
}

cl::Event ResourceManager::completeEvent()
{
    cl::UserEvent event(context);
    event.setStatus(CL_COMPLETE);
    return event;
}

cl::Kernel ResourceManager::createKernel(const std::string& source, 
                                          const std::string& kernelName)
{
    return createKernels(source, std::vector<std::string>(1,kernelName)).front();
}

std::vector<cl::Kernel> ResourceManager::createKernelsFromFile(const std::string& fileName, 
                                                               const std::vector<std::string>& kernelNames)
{
    std::ifstream file(fileName.c_str(), std::ios::in);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open source file.");
    }
    std::ostringstream source;
    source << file.rdbuf();
    return createKernels(source.str(), kernelNames);
}

std::vector<cl::Kernel> ResourceManager::createKernels(const std::string& source, 
                                                       const std::vector<std::string>& kernelNames)
{
#ifdef STATS
    timeval start, end;
    gettimeofday(&start,NULL);
#endif

#ifdef DEBUG
    std::cout << "Program build :\n";
    std::cout << "------------------- SOURCE -----------------------\n";
    std::cout << source;
    std::cout << "------------------ SOURCE END --------------------" << std::endl;
#endif
    cl::Program::Sources sources(1,std::make_pair(source.c_str(),source.size()));
    cl::Program program(context, sources);
    try {
        program.build(devices);
    } catch (cl::Error) {
#ifdef DEBUG
        std::cerr << "Program build error:\n";
        std::cerr << "------------------- SOURCE -----------------------\n";
        std::cerr << source;
        std::cerr << "------------------ SOURCE END --------------------\n";
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
#endif
        throw std::runtime_error("Could not build Kernel.");
    }
    
    std::vector<cl::Kernel> kernels;
    for (std::vector<std::string>::const_iterator knit = kernelNames.begin(); knit != kernelNames.end(); ++knit)
    {
        kernels.push_back(cl::Kernel(program, knit->c_str()));
    }
#ifdef STATS
    gettimeofday(&end,NULL);
    resourceCreateKernel += (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_usec - start.tv_usec);
#endif
    return kernels;
}

cl::Event ResourceManager::enqueueNDRangeKernel(const cl::Kernel& kernel, 
                                                const cl::NDRange& globalSize,
                                                const cl::NDRange& localSize,
                                                const std::vector<cl::Event>* waitFor,
                                                unsigned int device)
{
    cl::Event event;
    try 
    {
        commandQueues[device].enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, waitFor, &event);
    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        throw err;
    }
#ifdef STATS
    event.setCallback(CL_COMPLETE, &eventProfiler, &resourceKernelExecute);
#endif
    return event;
}

std::vector<size_t> ResourceManager::localShape(const std::vector<size_t>& globalShape)
{
    switch (globalShape.size())
    {
    case 1:
        return localShape1D; 
    case 2:
        return localShape2D; 
    case 3:
        return localShape3D; 
    default:
        assert (false);
    }
}

#ifdef STATS
void CL_CALLBACK ResourceManager::eventProfiler(cl_event ev, cl_int eventStatus, void* total)
{
    assert(eventStatus == CL_COMPLETE);
    cl::Event event(ev);
    cl_ulong start, end;
    start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    end =  event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    *(double*)total += (double)(end - start) / 1000.0;
}
#endif

std::string ResourceManager::getKernelPath()
{
    char* dir = cphvb_component_config_lookup(component, "ocldir");
    if (dir == NULL)
        return std::string("/opt/cphvb/lib/ocl_source");
    return std::string(dir);
}

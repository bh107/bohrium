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

#ifndef __RESOURCEMANAGER_HPP
#define __RESOURCEMANAGER_HPP

#include <CL/cl.hpp>
#include <vector>
#include <map>
#include <cphvb.h>
#ifdef STATS
#include "timing.h"
#endif

class ResourceManager
{
private:
    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::CommandQueue> commandQueues;
    typedef std::multimap<size_t,cl::Buffer*> BufferCache;
    BufferCache bufferCache; 
    size_t maxWorkGroupSize;
    cl_uint maxWorkItemDims;
    std::vector<size_t> maxWorkItemSizes;
    cphvb_component* component;
    std::vector<size_t> localShape1D;
    std::vector<size_t> localShape2D;
    std::vector<size_t> localShape3D;
public:
#ifdef STATS
    double batchBuild;
    double batchSource;
    double resourceCreateKernel;
    double resourceBufferWrite;
    double resourceBufferRead;
    double resourceKernelExecute;
    static void CL_CALLBACK eventProfiler(cl_event event, cl_int eventStatus, void* total);
#endif
    ResourceManager(cphvb_component* _component);
    ~ResourceManager();
    cl::Buffer* newBuffer(size_t size);
    void bufferDone(cl::Buffer* buffer);
    cl::Buffer createBuffer(size_t size);
    // We allways read synchronous with at most one event to wait for.
    // Because we are handing off the array
    void readBuffer(const cl::Buffer* buffer,
                    void* hostPtr, 
                    cl::Event waitFor,
                    unsigned int device);
    // We allways write asynchronous with NO events to wait for.
    // Because we just recieved the array from upstream
    cl::Event enqueueWriteBuffer(cl::Buffer* buffer,
                                 const void* hostPtr, 
                                 std::vector<cl::Event> waitFor, 
                                 unsigned int device);
    cl::Event completeEvent();
    cl::Kernel createKernel(const std::string& source, 
                            const std::string& kernelName);
    std::vector<cl::Kernel> createKernelsFromFile(const std::string& fileName, 
                                                  const std::vector<std::string>& kernelNames);
    std::vector<cl::Kernel> createKernels(const std::string& source, 
                                          const std::vector<std::string>& kernelNames);
    cl::Event enqueueNDRangeKernel(const cl::Kernel& kernel, 
                                   const cl::NDRange& globalSize,
                                   const cl::NDRange& localSize,
                                   const std::vector<cl::Event>* waitFor,
                                   unsigned int device);
    std::vector<size_t> localShape(const std::vector<size_t>& globalShape);
    std::string getKernelPath();
};

#endif


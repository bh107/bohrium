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

class ResourceManager
{
private:
    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::CommandQueue> commandQueues;
public:
    ResourceManager();
    cl::Buffer createBuffer(size_t size);
    cl::Event enqueueReadBuffer(const cl::Buffer buffer,
                                void* hostPtr, 
                                const std::vector<cl::Event>* waitFor,
                                int device);
    cl::Event enqueueWriteBuffer(const cl::Buffer buffer,
                                 const void* hostPtr, 
                                 const std::vector<cl::Event>* waitFor,
                                 int device);
    cl::Event completeEvent();
    cl::Kernel createKernel(const char* source, const char* kernelName);
};

#endif


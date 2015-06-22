/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __RESOURCEMANAGER_HPP
#define __RESOURCEMANAGER_HPP

#include "cl.hpp"
#include <vector>
#include <map>
#include <bh.h>
#define BH_TIMING_SUM
#include <bh_timing.hpp>
#include "OCLtype.h"

class ResourceManager
{
private:
    bh_component* component;
    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::CommandQueue> commandQueues;
    size_t maxWorkGroupSize;
    cl_uint maxWorkItemDims;
    std::vector<size_t> maxWorkItemSizes;
    std::vector<size_t> localShape1D;
    std::vector<size_t> localShape2D;
    std::vector<size_t> localShape3D;
    OCLtype _intpType;
    bool _float64;
    bool _fixedSizeKernel;
    bool _dynamicSizeKernel;
    bool _asyncCompile;
    bool _verbose;
    bool _printSource;
    bool _timing; 
    std::string compilerOptions;
    void registerExtensions(const std::vector<std::string>& extensions);
    bool setContext(const std::vector<cl::Platform>& platforms, cl_device_type device_type);
public:
    bh::Timer<>* codeGen;
    bh::Timer<>* kernelGen;
    bh::Timer<bh::timing4,1000000000>* bufferWrite;
    bh::Timer<bh::timing4,1000000000>* bufferRead;
    bh::Timer<bh::timing4,1000000000>* kernelExec;
    ~ResourceManager();
    static void CL_CALLBACK eventProfiler(cl::Event event, cl_int eventStatus, void* total);
    ResourceManager(bh_component* _component);
    cl::Buffer* createBuffer(size_t size);
    // We allways read synchronous with at most one event to wait for.
    // Because we are handing off the array
    void readBuffer(const cl::Buffer& buffer,
                    void* hostPtr, 
                    cl::Event waitFor,
                    unsigned int device);
    // We allways write asynchronous with NO events to wait for.
    // Because we just recieved the array from upstream
    cl::Event enqueueWriteBuffer(const cl::Buffer& buffer,
                                 const void* hostPtr, 
                                 std::vector<cl::Event> waitFor, 
                                 unsigned int device);
    cl::Event completeEvent();
    cl::Kernel createKernel(const std::string& source, 
                            const std::string& kernelName,
                            const std::string& options = std::string(""));
    std::vector<cl::Kernel> createKernelsFromFile(const std::string& fileName, 
                                                  const std::vector<std::string>& kernelNames,
                                                  const std::string& options = std::string(""));
    std::vector<cl::Kernel> createKernels(const std::string& source, 
                                          const std::vector<std::string>& kernelNames, 
                                          const std::string& options = std::string(""));
    cl::Event enqueueNDRangeKernel(const cl::Kernel& kernel, 
                                   const cl::NDRange& globalSize,
                                   const cl::NDRange& localSize,
                                   const std::vector<cl::Event>* waitFor,
                                   unsigned int device);
    std::vector<size_t> localShape(const std::vector<size_t>& globalShape);
    bh_error childExecute(bh_ir* bhir);
    OCLtype intpType() const { return _intpType; }
    bool float64() const { return _float64; }
    bool fixedSizeKernel() const { return _fixedSizeKernel; }
    bool dynamicSizeKernel() const { return _dynamicSizeKernel; }
    bool asyncCompile() const { return _asyncCompile; }
    bool verbose() const { return _verbose; }
    bool timing() const { return _timing; }
    bool printSource() const { return _printSource; }
};

#endif

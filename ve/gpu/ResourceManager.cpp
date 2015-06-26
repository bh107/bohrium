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

ResourceManager::ResourceManager(bh_component* _component) 
    : component(_component)
    , _fixedSizeKernel(true)
    , _dynamicSizeKernel(true)
    , _asyncCompile(true)

{
    _verbose = bh_component_config_lookup_bool(component, "verbose", 0);
    _timing = bh_component_config_lookup_bool(component, "timing", 0);
    _printSource = bh_component_config_lookup_bool(component, "print_source", 0);
    bool forceCPU  = bh_component_config_lookup_bool(component, "force_cpu", 0);
    char* dir = bh_component_config_lookup(component, "include");
    if (dir == NULL)
        compilerOptions = std::string("-I/opt/bohrium/gpu/include");
    else
        compilerOptions = std::string("-I") + std::string(dir);

    char* compiler_options = bh_component_config_lookup(component, "compiler_options");
    if (compiler_options != NULL)
    {
        compilerOptions += std::string(" ") + std::string(compiler_options);
        if (_verbose)
            std::cout << "[Info] [GPU] Compiler options: " << compiler_options << std::endl;

    }
    char* kernel_type = bh_component_config_lookup(component, "kernel");
    if (kernel_type != NULL)
    {
        std::string kernelType(kernel_type);
        if (kernelType.find("fixed") != std::string::npos)
            _dynamicSizeKernel = false;
        if (kernelType.find("dynamic") != std::string::npos)
            _fixedSizeKernel = false;
    }
    if (_verbose)
        std::cout << "[Info] [GPU] Kernel type: " <<
            (_dynamicSizeKernel&&_fixedSizeKernel?"both":(_dynamicSizeKernel?"dynamic":"fixed")) << std::endl;

    char* compile_type = bh_component_config_lookup(component, "compile");
    if (compile_type != NULL)
    {
        std::string compileType(compile_type);
        if (compileType.find("sync") != std::string::npos && compileType.find("async") == std::string::npos)
            _asyncCompile = false;
    }
    if (_verbose)
        std::cout << "[Info] [GPU] Compile type: " << (_asyncCompile?"async.":"sync.") << std::endl;

    localShape1D.push_back(bh_component_config_lookup_int(component, "work_goup_size_1dx",128));
    localShape2D.push_back(bh_component_config_lookup_int(component, "work_goup_size_2dx",32));
    localShape2D.push_back(bh_component_config_lookup_int(component, "work_goup_size_2dy",4));
    localShape3D.push_back(bh_component_config_lookup_int(component, "work_goup_size_3dx",32));
    localShape3D.push_back(bh_component_config_lookup_int(component, "work_goup_size_3dy",2));
    localShape3D.push_back(bh_component_config_lookup_int(component, "work_goup_size_3dz",2));

    if (_verbose)
        std::cout << "[Info] [GPU] Work group sizes: 1D[" << localShape1D[0] << "], 2D[" <<
            localShape2D[0] << ", " << localShape2D[1] << "], 3D[" << localShape3D[0] <<
            ", " << localShape3D[1] << ", " << localShape3D[2] << "]" << std::endl;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (forceCPU || !setContext(platforms,CL_DEVICE_TYPE_GPU))
    {
        if (!setContext(platforms,CL_DEVICE_TYPE_CPU))
            throw std::runtime_error("Could not find valid OpenCL platform.");
        else
            std::cerr << "[GPU-VE] Unable to find GPU running on CPU. ONLY FOR TESTING PURPOSES'" << std::endl;
    }
    std::vector<std::string> extensions;
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    for(std::vector<cl::Device>::iterator dit = devices.begin(); dit != devices.end(); ++dit)
    {
        cl_command_queue_properties properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        if (_timing)
            properties |=  CL_QUEUE_PROFILING_ENABLE;
        commandQueues.push_back(cl::CommandQueue(context,*dit,properties));
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
        extensions.push_back(dit->getInfo<CL_DEVICE_EXTENSIONS>());
    }
    if (devices[0].getInfo<CL_DEVICE_ADDRESS_BITS>() == 64)
    {
        _intpType = OCL_INT64;
    } else {
        _intpType = OCL_INT32;
    }
    registerExtensions(extensions);

    if (_timing)
    {
        codeGen = new bh::Timer<>("[GPU] Code generation");
        kernelGen = new bh::Timer<>("[GPU] Kernel generation");
        bufferWrite = new bh::Timer<bh::timing4,1000000000>("[GPU] Writing buffers");
        bufferRead = new bh::Timer<bh::timing4,1000000000>("[GPU] Reading buffers");
        kernelExec = new bh::Timer<bh::timing4,1000000000>("[GPU] Kernel execution");
    }
}

bool ResourceManager::setContext(const std::vector<cl::Platform>& platforms, cl_device_type device_type)
{
    bool found = false;
    for(const cl::Platform platform: platforms)
    {
        try {
            cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),0};
            context = cl::Context(device_type, props);
            found = true;
            break;
        }
        catch (cl::Error)
        {
            found = false;
        }
    }
    return found;
}

ResourceManager::~ResourceManager()
{
    if (_timing)
    {
        delete codeGen;
        delete kernelGen;
        delete bufferWrite;
        delete bufferRead;
        delete kernelExec;
    }
}

void ResourceManager::registerExtensions(const std::vector<std::string>& extensions)
{
    _float64 = extensions[0].find("cl_khr_fp64") != std::string::npos;
    if (_verbose)
        std::cout << "[Info] [GPU] float64 support: " << (_float64?"true":"false") << std::endl;
}

cl::Buffer* ResourceManager::createBuffer(size_t size)
{
    return new cl::Buffer(context, CL_MEM_READ_WRITE, size);
}

void ResourceManager::readBuffer(const cl::Buffer& buffer,
                                 void* hostPtr, 
                                 cl::Event waitFor,
                                 unsigned int device)
{
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    std::vector<cl::Event> readerWaitFor(1,waitFor);
    cl::Event event;
    try {
        commandQueues[device].enqueueReadBuffer(buffer, CL_TRUE, 0, size, hostPtr, &readerWaitFor, 
        (_timing?&event:NULL));
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueReadBuffer: \"" << e.err() << "\"" << std::endl;
    }
    if (_timing)
        event.setCallback(CL_COMPLETE, &eventProfiler, bufferRead);
}

cl::Event ResourceManager::enqueueWriteBuffer(const cl::Buffer& buffer,
                                              const void* hostPtr, 
                                              std::vector<cl::Event> waitFor, 
                                              unsigned int device)
{
    cl::Event event;
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    try {
        commandQueues[device].enqueueWriteBuffer(buffer, CL_FALSE, 0, size, hostPtr, &waitFor, &event);
    } catch (cl::Error e) {
        std::cerr << "[VE-GPU] Could not enqueueWriteBuffer: \"" << e.what() << "\"" << std::endl;
        throw e;
    }
    if (_timing)
        event.setCallback(CL_COMPLETE, &eventProfiler, bufferWrite);
    return event;
}

cl::Event ResourceManager::completeEvent()
{
    cl::UserEvent event(context);
    event.setStatus(CL_COMPLETE);
    return event;
}

cl::Kernel ResourceManager::createKernel(const std::string& source, 
                                         const std::string& kernelName,
                                         const std::string& options)
{
    return createKernels(source, std::vector<std::string>(1,kernelName), options).front();
}

std::vector<cl::Kernel> ResourceManager::createKernelsFromFile(const std::string& fileName, 
                                                               const std::vector<std::string>& kernelNames,
                                                               const std::string& options)
{
    std::ifstream file(fileName.c_str(), std::ios::in);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open source file.");
    }
    std::ostringstream source;
    source << file.rdbuf();
    return createKernels(source.str(), kernelNames, options);
}

std::vector<cl::Kernel> ResourceManager::createKernels(const std::string& source, 
                                                       const std::vector<std::string>& kernelNames,
                                                       const std::string& options)
{

    bh_uint64 start;
    if (_timing)
        start = bh::Timer<>::stamp(); 

    std::string compilerOptions(this->compilerOptions);
    compilerOptions += std::string(" ") + options;
    if (_printSource)
    {
        std::cout << "Program build :\n";
        std::cout << "Options :" << compilerOptions << "\n";
        std::cout << "------------------- SOURCE -----------------------\n";
        std::cout << source;
        std::cout << "------------------ SOURCE END --------------------" << std::endl;
    }
    cl::Program::Sources sources(1,std::make_pair(source.c_str(),source.size()));
    cl::Program program(context, sources);
    try {
        program.build(devices,(compilerOptions).c_str());
    } catch (cl::Error) {
        std::cerr << "Program build error:\n";
        std::cout << "Options :" << compilerOptions << "\n";
        std::cerr << "------------------- SOURCE -----------------------\n";
        std::cerr << source;
        std::cerr << "------------------ SOURCE END --------------------\n";
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        throw std::runtime_error("Could not build Kernel.");
    }
    std::vector<cl::Kernel> kernels;
    for (std::vector<std::string>::const_iterator knit = kernelNames.begin(); knit != kernelNames.end(); ++knit)
    {
        try {
            kernels.push_back(cl::Kernel(program, knit->c_str()));
        } catch (cl::Error e) {
            std::cerr << "Could not create cl::Kernel " <<  knit->c_str() << ": " << e.what() << " " << 
                e.err() << std::endl;  
        }
    }
    if (_timing)
        kernelGen->add({start, bh::Timer<>::stamp()});
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
    if (_timing)
    event.setCallback(CL_COMPLETE, &eventProfiler, kernelExec);
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

void CL_CALLBACK ResourceManager::eventProfiler(cl::Event event, cl_int eventStatus, void* timer)
{
    assert(eventStatus == CL_COMPLETE);
    ((bh::Timer<bh::timing4,1000000000>*)timer)->add({ event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>(),
                event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>(),
                event.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
                event.getProfilingInfo<CL_PROFILING_COMMAND_END>()});
}

bh_error ResourceManager::childExecute(bh_ir* bhir)
{
    bh_error err = BH_ERROR;
    for (int i = 0; i < component->nchildren; ++i)
    {
        bh_component_iface* child = &component->children[i];
        err = child->execute(bhir);
        if (err == BH_SUCCESS)
            break;
    }
    return err;
}

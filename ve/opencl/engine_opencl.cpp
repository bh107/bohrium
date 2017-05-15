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

#include <vector>
#include <iostream>
#include <boost/functional/hash.hpp>

#include <jitk/kernel.hpp>

#include "engine_opencl.hpp"

using namespace std;

namespace {

#define CL_DEVICE_AUTO 1024 // More than maximum in the bitmask

map<const string, cl_device_type> device_map = {
    {"auto",        CL_DEVICE_AUTO},
    {"gpu",         CL_DEVICE_TYPE_GPU},
    {"accelerator", CL_DEVICE_TYPE_ACCELERATOR},
    {"default",     CL_DEVICE_TYPE_DEFAULT},
    {"cpu",         CL_DEVICE_TYPE_CPU}
};

// Get the OpenCL device (search order: GPU, ACCELERATOR, DEFAULT, and CPU)
cl::Device getDevice(const cl::Platform &platform, const string &default_device_type) {
    vector<cl::Device> device_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);

    if(device_list.size() == 0){
        throw runtime_error("No OpenCL device found");
    }

    if (!util::exist(device_map, default_device_type)) {
        stringstream ss;
        ss << "'" << default_device_type << "' is not a OpenCL device type. " \
           << "Must be one of 'auto', 'gpu', 'accelerator', 'cpu', or 'default'";
        throw runtime_error(ss.str());
    } else if (device_map[default_device_type] != CL_DEVICE_AUTO) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_map[default_device_type]) == device_map[default_device_type]) {
                return device;
            }
        }
        stringstream ss;
        ss << "Could not find selected OpenCL device type ('" \
           << default_device_type << "') on default platform";
        throw runtime_error(ss.str());
    }

    // Type was 'auto'
    for (auto &device_type: device_map) {
        for (auto &device: device_list) {
            if ((device.getInfo<CL_DEVICE_TYPE>() & device_type.second) == device_type.second) {
                return device;
            }
        }
    }

    throw runtime_error("No OpenCL device of usable type found");
}
}

namespace bohrium {

static boost::hash<string> hasher;

EngineOpenCL::EngineOpenCL(const ConfigParser &config, jitk::Statistics &stat) :
                                    work_group_size_1dx(config.defaultGet<int>("work_group_size_1dx", 128)),
                                    work_group_size_2dx(config.defaultGet<int>("work_group_size_2dx", 32)),
                                    work_group_size_2dy(config.defaultGet<int>("work_group_size_2dy", 4)),
                                    work_group_size_3dx(config.defaultGet<int>("work_group_size_3dx", 32)),
                                    work_group_size_3dy(config.defaultGet<int>("work_group_size_3dy", 2)),
                                    work_group_size_3dz(config.defaultGet<int>("work_group_size_3dz", 2)),
                                    compile_flg(config.defaultGet<string>("compiler_flg", "")),
                                    default_device_type(config.defaultGet<string>("device_type", "auto")),
                                    platform_no(config.defaultGet<int>("platform_no", -1)),
                                    verbose(config.defaultGet<bool>("verbose", false)),
                                    stat(stat),
                                    prof(config.defaultGet<bool>("prof", false))
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0) {
        throw runtime_error("No OpenCL platforms found");
    }

    bool found = false;
    cl::Platform platform;
    if (platform_no == -1) {
        for (auto pform : platforms) {
            // Pick first valid platform
            try {
                // Get the device of the platform
                platform = pform;
                device = getDevice(platform, default_device_type);
                found = true;
            } catch(cl::Error err) {
                // We try next platform
            }
        }
    } else {
        if (platform_no > ((int) platforms.size()-1)) {
            std::stringstream ss;
            ss << "No such OpenCL platform. Tried to fetch #";
            ss << platform_no << " out of ";
            ss << platforms.size()-1 << "." << endl;
            throw std::runtime_error(ss.str());
        }

        platform = platforms[platform_no];
        device = getDevice(platform, default_device_type);
        found = true;
    }

    if (verbose) {
        cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
    }

    if (!found) {
        throw runtime_error("Invalid OpenCL device/platform");
    }

    if(verbose) {
        cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() \
             << " ("<< device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << ")" << endl;
    }

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);
}


pair<cl::NDRange, cl::NDRange> EngineOpenCL::NDRanges(const vector<const jitk::LoopB*> &threaded_blocks) const {
    const auto &b = threaded_blocks;
    switch (b.size()) {
        case 1:
        {
            const cl_ulong lsize = work_group_size_1dx;
            const cl_ulong rem = b[0]->size % lsize;
            const cl_ulong gsize = b[0]->size + (rem==0?0:(lsize-rem));
            return make_pair(cl::NDRange(gsize), cl::NDRange(lsize));
        }
        case 2:
        {
            const cl_ulong lsize_x = work_group_size_2dx;
            const cl_ulong lsize_y = work_group_size_2dy;
            const cl_ulong rem_x = b[0]->size % lsize_x;
            const cl_ulong rem_y = b[1]->size % lsize_y;
            const cl_ulong gsize_x = b[0]->size + (rem_x==0?0:(lsize_x-rem_x));
            const cl_ulong gsize_y = b[1]->size + (rem_y==0?0:(lsize_y-rem_y));
            return make_pair(cl::NDRange(gsize_x, gsize_y), cl::NDRange(lsize_x, lsize_y));
        }
        case 3:
        {
            const cl_ulong lsize_x = work_group_size_3dx;
            const cl_ulong lsize_y = work_group_size_3dy;
            const cl_ulong lsize_z = work_group_size_3dz;
            const cl_ulong rem_x = b[0]->size % lsize_x;
            const cl_ulong rem_y = b[1]->size % lsize_y;
            const cl_ulong rem_z = b[2]->size % lsize_z;
            const cl_ulong gsize_x = b[0]->size + (rem_x==0?0:(lsize_x-rem_x));
            const cl_ulong gsize_y = b[1]->size + (rem_y==0?0:(lsize_y-rem_y));
            const cl_ulong gsize_z = b[2]->size + (rem_z==0?0:(lsize_z-rem_z));
            return make_pair(cl::NDRange(gsize_x, gsize_y, gsize_z), cl::NDRange(lsize_x, lsize_y, lsize_z));
        }
        default:
            throw runtime_error("NDRanges: maximum of three dimensions!");
    }
}

void EngineOpenCL::execute(const std::string &source, const jitk::Kernel &kernel,
                           const vector<const jitk::LoopB*> &threaded_blocks,
                           const vector<const bh_view*> &offset_strides,
                           const vector<const bh_instruction*> &constants) {
    size_t hash = hasher(source);
    ++stat.kernel_cache_lookups;
    cl::Program program;

    auto tcompile = chrono::steady_clock::now();

    // Do we have the program already?
    if (_programs.find(hash) != _programs.end()) {
        program = _programs.at(hash);
    } else {
        // Or do we have to compile it
        ++stat.kernel_cache_misses;
        program = cl::Program(context, source);
        try {
            if (verbose) {
                cout << "********** Compile Flags **********" << endl \
                << compile_flg.c_str() << endl \
                << "************ Flags END ************" << endl << endl;
                cout << "************ Build Log ************" << endl \
                 << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) \
                 << "^^^^^^^^^^^^^ Log END ^^^^^^^^^^^^^" << endl << endl;
            }
            program.build({device}, compile_flg.c_str());
        } catch (cl::Error e) {
            cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
            throw;
        }
        _programs[hash] = program;
    }
    stat.time_compile += chrono::steady_clock::now() - tcompile;

    // Let's execute the OpenCL kernel
    cl::Kernel opencl_kernel = cl::Kernel(program, "execute");

    cl_uint i = 0;
    for (bh_base *base: kernel.getNonTemps()) { // NB: the iteration order matters!
        opencl_kernel.setArg(i++, *getBuffer(base));
    }

    for (const bh_view *view: offset_strides) {
        uint64_t t1 = (uint64_t) view->start;
        opencl_kernel.setArg(i++, t1);
        for (int j=0; j<view->ndim; ++j) {
            uint64_t t2 = (uint64_t) view->stride[j];
            opencl_kernel.setArg(i++, t2);
        }
    }

    for (const bh_instruction *instr: constants) {
        switch (instr->constant.type) {
            case BH_BOOL:
                opencl_kernel.setArg(i++, instr->constant.value.bool8);
                break;
            case BH_INT8:
                opencl_kernel.setArg(i++, instr->constant.value.int8);
                break;
            case BH_INT16:
                opencl_kernel.setArg(i++, instr->constant.value.int16);
                break;
            case BH_INT32:
                opencl_kernel.setArg(i++, instr->constant.value.int32);
                break;
            case BH_INT64:
                opencl_kernel.setArg(i++, instr->constant.value.int64);
                break;
            case BH_UINT8:
                opencl_kernel.setArg(i++, instr->constant.value.uint8);
                break;
            case BH_UINT16:
                opencl_kernel.setArg(i++, instr->constant.value.uint16);
                break;
            case BH_UINT32:
                opencl_kernel.setArg(i++, instr->constant.value.uint32);
                break;
            case BH_UINT64:
                opencl_kernel.setArg(i++, instr->constant.value.uint64);
                break;
            case BH_FLOAT32:
                opencl_kernel.setArg(i++, instr->constant.value.float32);
                break;
            case BH_FLOAT64:
                opencl_kernel.setArg(i++, instr->constant.value.float64);
                break;
            case BH_COMPLEX64:
                opencl_kernel.setArg(i++, instr->constant.value.complex64);
                break;
            case BH_COMPLEX128:
                opencl_kernel.setArg(i++, instr->constant.value.complex128);
                break;
            default:
                std::cerr << "Unknown OpenCL type: " << bh_type_text(instr->constant.type) << std::endl;
                throw std::runtime_error("Unknown OpenCL type");
        }
    }

    const auto ranges = NDRanges(threaded_blocks);
    auto texec = chrono::steady_clock::now();
    queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);
    queue.finish();
    stat.time_exec += chrono::steady_clock::now() - texec;
}

void EngineOpenCL::set_constructor_flag(std::vector<bh_instruction*> &instr_list) {
    jitk::util_set_constructor_flag(instr_list, buffers);
}

} // bohrium

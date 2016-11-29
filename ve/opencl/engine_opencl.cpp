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

namespace bohrium {

//static boost::hash<string> hasher;

EngineOpenCL::EngineOpenCL(const ConfigParser &config) :
                                compile_flg(config.defaultGet<string>("compiler_flg", "")),
                                verbose(config.defaultGet<bool>("verbose", false)) {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0) {
        throw runtime_error("No OpenCL platforms found");
    }
    cl::Platform default_platform=platforms[0];
    cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

    //get default device of the default platform
    vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.size()==0){
        throw runtime_error("No OpenCL device found");
    }
    default_device = devices[0];
    cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

    vector<cl::Device> dev_list = {default_device};
    context = cl::Context(dev_list);
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
                           const vector<const jitk::LoopB*> &threaded_blocks, cl::CommandQueue &queue) {

    cl::Program program(context, source);
    try {
        program.build({default_device}, compile_flg.c_str());
        if (verbose) {
            cout << "************ Build Log ************" << endl \
                 << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) \
                 << "^^^^^^^^^^^^^ Log END ^^^^^^^^^^^^^" << endl << endl;
        }
    } catch (cl::Error e) {
        cerr << "Error building: " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
        throw;
    }

    // Let's execute the OpenCL kernel
    cl::Kernel opencl_kernel = cl::Kernel(program, "execute");
    {
        cl_uint i = 0;
        for (bh_base *base: kernel.getNonTemps()) { // NB: the iteration order matters!
            opencl_kernel.setArg(i++, *buffers.at(base));
        }
    }
    const auto ranges = NDRanges(threaded_blocks);
    queue.enqueueNDRangeKernel(opencl_kernel, cl::NullRange, ranges.first, ranges.second);
    queue.finish();
}

} // bohrium


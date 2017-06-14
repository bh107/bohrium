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

#ifndef __BH_VE_OPENCL_ENGINE_OPENCL_HPP
#define __BH_VE_OPENCL_ENGINE_OPENCL_HPP

#include <map>
#include <memory>
#include <vector>
#include <tuple>
#include <chrono>
#include <boost/filesystem.hpp>

#include <bh_config_parser.hpp>
#include <bh_view.hpp>
#include <jitk/statistics.hpp>
#include <jitk/kernel.hpp>
#include <jitk/codegen_util.hpp>

#include <cuda.h>

#include "compiler.hpp"

namespace {
    // This will output the proper CUDA error strings
    // in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( CUresult err, const char *file, const int line )
    {
        if( CUDA_SUCCESS != err) {
            fprintf(stderr,
                    "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                    err, file, line );
            exit(-1);
        }
    }
}

namespace bohrium {

class EngineCUDA {
private:
    // Map of all compiled OpenCL programs
    std::map<uint64_t, CUfunction> _programs;
    // The CUDA context and device used throughout the execution
    CUdevice   device;
    CUcontext  context;
    // OpenMP work group sizes
    const uint64_t work_group_size_1dx;
    const uint64_t work_group_size_2dx;
    const uint64_t work_group_size_2dy;
    const uint64_t work_group_size_3dx;
    const uint64_t work_group_size_3dy;
    const uint64_t work_group_size_3dz;
    // OpenCL compile flags
    const std::string compile_flg;
    // Default device type
    const std::string default_device_type;
    // Default platform number
    const int platform_no;
    // Returns the global and local work OpenCL ranges based on the 'threaded_blocks'
    //std::pair<cl::NDRange, cl::NDRange> NDRanges(const std::vector<const jitk::LoopB*> &threaded_blocks) const;
    // A map of allocated buffers on the device
    std::map<bh_base*, CUdeviceptr> buffers;
    // Verbose flag
    const bool verbose;
    // Some statistics
    jitk::Statistics &stat;
    // Record profiling statistics
    const bool prof;

    // Path to a temporary directory for the source and object files
    const boost::filesystem::path tmp_dir;

    // Path to the directory of the source files
    const boost::filesystem::path source_dir;

    // Path to the directory of the object files
    const boost::filesystem::path object_dir;

    // The compiler to use when function doesn't exist
    const Compiler compiler;

    // Returns the block and thread sizes based on the 'threaded_blocks'
    std::pair<std::tuple<uint32_t, uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t> >
        NDRanges(const std::vector<const jitk::LoopB*> &threaded_blocks) const;

public:
    EngineCUDA(const ConfigParser &config, jitk::Statistics &stat);
    ~EngineCUDA() {
        cuCtxDetach(context);
    }

    // Execute the 'source'
    void execute(const std::string &source, const jitk::Kernel &kernel,
                 const std::vector<const jitk::LoopB*> &threaded_blocks,
                 const std::vector<const bh_view*> &offset_strides,
                 const std::vector<const bh_instruction*> &constants);

    // Delete a buffer
    template <typename T>
    void delBuffer(T &base) {
        checkCudaErrors(cuMemFree(buffers[base]));
        buffers.erase(base);
    }

    // Retrieve a single buffer
    template <typename T>
    CUdeviceptr *getBuffer(T &base) {
        if(buffers.find(base) == buffers.end()) {
            std::vector<T> vec = {base};
            copyToDevice(vec);
        }
        return &buffers[base];
    }

    // Copy 'bases' to the host (ignoring bases that isn't on the device)
    template <typename T>
    void copyToHost(T &bases) {
        auto tcopy = std::chrono::steady_clock::now();
        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: bases) {
            if (buffers.find(base) != buffers.end()) {
                bh_data_malloc(base);
                if (verbose) {
                    std::cout << "Copy to host: " << *base << std::endl;
                }
                checkCudaErrors(cuMemcpyDtoH(base->data, buffers.at(base), bh_base_size(base)));
                // When syncing we assume that the host writes to the data and invalidate the device data thus
                // we have to remove its data buffer
                delBuffer(base);
            }
        }
        stat.time_copy2host += std::chrono::steady_clock::now() - tcopy;
    }

    // Copy 'base_list' to the device (ignoring bases that is already on the device)
    template <typename T>
    void copyToDevice(T &base_list) {

        // Let's update the maximum memory usage on the device
        if (prof) {
            uint64_t sum = 0;
            for (const auto &b: buffers) {
                sum += bh_base_size(b.first);
            }
            stat.max_memory_usage = sum > stat.max_memory_usage?sum:stat.max_memory_usage;
        }

        auto tcopy = std::chrono::steady_clock::now();
        for(bh_base *base: base_list) {
            if (buffers.find(base) == buffers.end()) { // We shouldn't overwrite existing buffers
                CUdeviceptr new_buf;
                checkCudaErrors(cuMemAlloc(&new_buf, bh_base_size(base)));
                buffers[base] = new_buf;

                // If the host data is non-null we should copy it to the device
                if (base->data != NULL) {
                    if (verbose) {
                        std::cout << "Copy to device: " << *base << std::endl;
                    }
                    checkCudaErrors(cuMemcpyHtoD(new_buf, base->data, bh_base_size(base)));
                }
            }
        }
        stat.time_copy2dev += std::chrono::steady_clock::now() - tcopy;
    }

    // Copy all bases to the host (ignoring bases that isn't on the device)
    void allBasesToHost() {
        std::vector<bh_base*> bases_on_device;
        for(auto &buf_pair: buffers) {
            bases_on_device.push_back(buf_pair.first);
        }
        copyToHost(bases_on_device);
    }

    // Sets the constructor flag of each instruction in 'instr_list'
    void set_constructor_flag(std::vector<bh_instruction*> &instr_list);
};

} // bohrium

#endif

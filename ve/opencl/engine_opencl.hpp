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
#include <chrono>

#include <bh_config_parser.hpp>
#include <bh_view.hpp>
#include <jitk/statistics.hpp>
#include <jitk/kernel.hpp>

#include "cl.hpp"

namespace bohrium {

class EngineOpenCL {
private:
    // Map of all compiled OpenCL programs
    std::map<uint64_t, cl::Program> _programs;
    // The OpenCL context, device, and queue used throughout the execution
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    // OpenCL work group sizes
    const cl_ulong work_group_size_1dx;
    const cl_ulong work_group_size_2dx;
    const cl_ulong work_group_size_2dy;
    const cl_ulong work_group_size_3dx;
    const cl_ulong work_group_size_3dy;
    const cl_ulong work_group_size_3dz;
    // OpenCL compile flags
    const std::string compile_flg;
    // Default device type
    const std::string default_device_type;
    // Default platform number
    const int platform_no;
    // Verbose flag
    const bool verbose;
    // Returns the global and local work OpenCL ranges based on the 'threaded_blocks'
    std::pair<cl::NDRange, cl::NDRange> NDRanges(const std::vector<const jitk::LoopB*> &threaded_blocks) const;
public:
    // A map of allocated buffers on the device
    std::map<bh_base*, std::unique_ptr<cl::Buffer> > buffers; //TODO; privatize
public:
    EngineOpenCL(const ConfigParser &config, jitk::Statistics &stat);

    // Some statistics
    jitk::Statistics &stat;

    // Execute the 'source'
    void execute(const std::string &source, const jitk::Kernel &kernel,
                 const std::vector<const jitk::LoopB*> &threaded_blocks,
                 const std::vector<const bh_view*> &offset_strides,
                 const std::vector<const bh_instruction*> &constants);

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
                queue.enqueueReadBuffer(*buffers.at(base), CL_FALSE, 0, (cl_ulong) bh_base_size(base), base->data);
                // When syncing we assume that the host writes to the data and invalidate the device data thus
                // we have to remove its data buffer
                buffers.erase(base);
            }
        }
        queue.finish();
        stat.time_copy2host += std::chrono::steady_clock::now() - tcopy;
    }

    // Copy 'base_list' to the device (ignoring bases that is already on the device)
    template <typename T>
    void copyToDevice(T &base_list) {
        auto tcopy = std::chrono::steady_clock::now();
        for(bh_base *base: base_list) {
            if (buffers.find(base) == buffers.end()) { // We shouldn't overwrite existing buffers
                cl::Buffer *buf = new cl::Buffer(context, CL_MEM_READ_WRITE, (cl_ulong) bh_base_size(base));
                buffers[base].reset(buf);

                // If the host data is non-null we should copy it to the device
                if (base->data != NULL) {
                    if (verbose) {
                        std::cout << "Copy to device: " << *base << std::endl;
                    }
                    queue.enqueueWriteBuffer(*buf, CL_FALSE, 0, (cl_ulong) bh_base_size(base), base->data);
                }
            }
        }
        queue.finish();
        stat.time_copy2dev += std::chrono::steady_clock::now() - tcopy;
    }

    template <typename T>
    cl::Buffer* getBuffer(T &base) {
        if(buffers.find(base) == buffers.end()) {
            std::vector<T> vec = {base};
            copyToDevice(vec);
        }
        return &(*buffers[base]);
    }

    // Get C buffer from wrapped C++ object
    template <typename T>
    cl_mem getCBuffer(T &base) {
        return (*getBuffer(base))();
    }

    cl::CommandQueue* getQueue() {
        return &queue;
    }

    // Get C command queue from wrapped C++ object
    cl_command_queue getCQueue() {
        return queue();
    }
};

} // bohrium

#endif

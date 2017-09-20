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
#include <jitk/codegen_util.hpp>

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
    // We save the OpenCL platform object for later information retrieval
    cl::Platform platform;
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
    // Returns the global and local work OpenCL ranges based on the 'threaded_blocks'
    std::pair<cl::NDRange, cl::NDRange> NDRanges(const std::vector<const jitk::LoopB*> &threaded_blocks) const;
    // A map of allocated buffers on the device
    std::map<bh_base*, std::unique_ptr<cl::Buffer> > buffers;
    // Verbose flag
    const bool verbose;
    // Some statistics
    jitk::Statistics &stat;
    // Record profiling statistics
    const bool prof;
    // Path to a temporary directory for the source and object files
    const boost::filesystem::path tmp_dir;
    // Path to the temporary directory of the source files
    const boost::filesystem::path tmp_src_dir;
    // Path to the temporary directory of the binary files (e.g. .clbin files)
    const boost::filesystem::path tmp_bin_dir;
    // Path to the directory of the cached binary files (e.g. .clbin files)
    const boost::filesystem::path cache_bin_dir;
    // The hash of the JIT compilation command
    size_t compilation_hash;
    // Return a kernel function based on the given 'source'
    cl::Program getFunction(const std::string &source);
public:
    EngineOpenCL(const ConfigParser &config, jitk::Statistics &stat);
    ~EngineOpenCL();

    // Execute the 'source'
    void execute(const std::string &source, const std::vector<bh_base*> &non_temps,
                 const std::vector<const jitk::LoopB*> &threaded_blocks,
                 const std::vector<const bh_view*> &offset_strides,
                 const std::vector<const bh_instruction*> &constants);

    // Check if `base` is on the device
    bool baseOnDevice(const bh_base *base) const {
        return util::exist_nconst(buffers, base);
    }

    // Copy 'bases' to the host (ignoring bases that isn't on the device)
    template <typename T>
    void copyToHost(T &bases) {
        auto tcopy = std::chrono::steady_clock::now();
        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: bases) {
            if (baseOnDevice(base)) {
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

    cl::Buffer *createBuffer(bh_base *base) {
        cl::Buffer *buf = new cl::Buffer(context, CL_MEM_READ_WRITE, (cl_ulong) bh_base_size(base));
        buffers[base].reset(buf);
        return buf;
    }

    cl::Buffer *createBuffer(bh_base *base, void* opencl_mem_ptr) {
        cl::Buffer *buf = new cl::Buffer();
        cl_mem _mem = reinterpret_cast<cl_mem>(opencl_mem_ptr);
        cl_int err = clRetainMemObject(_mem); // Increments the memory object reference count
        if (err != CL_SUCCESS) {
            throw std::runtime_error("OpenCL - clRetainMemObject(): failed");
        }
        (*buf) = _mem;
        buffers[base].reset(buf);
        return buf;
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
                cl::Buffer *buf = createBuffer(base);

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

    // Return a YAML string describing this component
    std::string info() const;

    // Retrieve a single buffer
    cl::Buffer* getBuffer(bh_base* base) {
        if(buffers.find(base) == buffers.end()) {
            std::vector<bh_base*> vec = {base};
            copyToDevice(vec);
        }
        return &(*buffers[base]);
    }

    // Delete a buffer
    void delBuffer(bh_base* base) {
        buffers.erase(base);
    }

    // Get C buffer from wrapped C++ object
    cl_mem getCBuffer(bh_base* base) {
        return (*getBuffer(base))();
    }

    // Get C context from wrapped C++ object
    cl_context getCContext() {
        return context();
    }

    // Get the OpenCL command queue object
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

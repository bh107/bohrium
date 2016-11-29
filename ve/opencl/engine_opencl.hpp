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

#include <bh_config_parser.hpp>
#include <bh_array.hpp>
#include "cl.hpp"

namespace bohrium {

class EngineOpenCL {
public:
    // The OpenCL context and device used throughout the execution
    cl::Context context;
    cl::Device default_device;
    // A map of allocated buffers on the device
    std::map<bh_base*, std::unique_ptr<cl::Buffer> > buffers;
    // OpenCL work group sizes
    cl_ulong work_group_size_1dx = 128;
    cl_ulong work_group_size_2dx = 32;
    cl_ulong work_group_size_2dy = 4;
    cl_ulong work_group_size_3dx = 32;
    cl_ulong work_group_size_3dy = 2;
    cl_ulong work_group_size_3dz = 2;
    // OpenCL compile flags
    const std::string compile_flg;
    // Verbose flag
    const bool verbose;
    // Returns the global and local work OpenCL ranges based on the 'threaded_blocks'
    std::pair<cl::NDRange, cl::NDRange> NDRanges(const std::vector<const jitk::LoopB*> &threaded_blocks) const;
public:
    EngineOpenCL(const ConfigParser &config);

    // Some statistics
    int64_t num_lookups = 0;
    int64_t num_lookup_misses = 0;

    // Execute the 'source'
    void execute(const std::string &source, const jitk::Kernel &kernel,
                 const std::vector<const jitk::LoopB*> &threaded_blocks,
                 cl::CommandQueue &queue);

    // Copy 'bases' to the host (ignoring bases that isn't on the device)
    template <typename T>
    void copyToHost(T &bases, cl::CommandQueue &queue) {
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
    }

};



} // bohrium

#endif

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
#pragma once

#include <map>
#include <memory>
#include <vector>
#include <chrono>

#include <bh_config_parser.hpp>
#include <bh_instruction.hpp>
#include <bh_component.hpp>
#include <bh_view.hpp>
#include <bh_malloc_cache.hpp>
#include <jitk/statistics.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/engines/engine_gpu.hpp>

#include "cl.hpp"

namespace bohrium {

class EngineOpenCL : public jitk::EngineGPU {
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

    // Returns the global and local work OpenCL ranges based on the 'thread_stack'
    std::pair<cl::NDRange, cl::NDRange> NDRanges(const std::vector<uint64_t> &thread_stack) const;

    // A map of allocated buffers on the device
    std::map<bh_base *, cl::Buffer *> buffers;

    // Return a kernel function based on the given 'source'
    cl::Program getFunction(const std::string &source);

    // We initialize the malloc cache with an alloc and free function
    MallocCache::FuncAllocT func_alloc = [this](uint64_t nbytes) -> void * {
        return (void *) new cl::Buffer(this->context, CL_MEM_READ_WRITE, (cl_ulong) nbytes);
    };
    MallocCache::FuncFreeT func_free = [](void *mem, uint64_t nbytes) { delete ((cl::Buffer *) mem); };
    MallocCache malloc_cache{func_alloc, func_free, 0};

public:
    EngineOpenCL(component::ComponentVE &comp, jitk::Statistics &stat);
    ~EngineOpenCL() override;

    // Execute the 'source'
    void execute(const jitk::SymbolTable &symbols,
                 const std::string &source,
                 uint64_t codegen_hash,
                 const std::vector<uint64_t> &thread_stack,
                 const std::vector<const bh_instruction *> &constants) override;

    // Copy 'bases' to the host (ignoring bases that isn't on the device)
    void copyToHost(const std::set<bh_base *> &bases) override;

    // Copy 'base_list' to the device (ignoring bases that is already on the device)
    void copyToDevice(const std::set<bh_base *> &base_list) override;

    // Sets the constructor flag of each instruction in 'instr_list'
    void setConstructorFlag(std::vector<bh_instruction *> &instr_list) override;

    // Copy all bases to the host (ignoring bases that isn't on the device)
    void copyAllBasesToHost() override;

    // Delete a buffer
    void delBuffer(bh_base *base) override;

    void writeKernel(const jitk::LoopB &kernel,
                     const jitk::SymbolTable &symbols,
                     const std::vector<uint64_t> &thread_stack,
                     uint64_t codegen_hash,
                     std::stringstream &ss) override;

    // Writes the OpenCL specific for-loop header
    void loopHeadWriter(const jitk::SymbolTable &symbols,
                        jitk::Scope &scope,
                        const jitk::LoopB &block,
                        const std::vector<uint64_t> &thread_stack,
                        std::stringstream &out) override;

    // Return a YAML string describing this component
    std::string info() const override;

    // Return OpenCL API types, which are used inside the JIT kernels
    const std::string writeType(bh_type dtype) override;

    cl::Buffer *createBuffer(bh_base *base) {
        auto *buf = reinterpret_cast<cl::Buffer *>(malloc_cache.alloc(base->nbytes()));
        bool inserted = buffers.insert(std::make_pair(base, buf)).second;
        if (not inserted) {
            throw std::runtime_error("OpenCL - createBuffer(): the base already has a buffer!");
        }
        return buf;
    }

    cl::Buffer *createBuffer(bh_base *base, void *opencl_mem_ptr) {
        auto *buf = new cl::Buffer();
        auto _mem = reinterpret_cast<cl_mem>(opencl_mem_ptr);
        cl_int err = clRetainMemObject(_mem); // Increments the memory object reference count
        if (err != CL_SUCCESS) {
            throw std::runtime_error("OpenCL - clRetainMemObject(): failed");
        }
        (*buf) = _mem;
        bool inserted = buffers.insert(std::make_pair(base, buf)).second;
        if (not inserted) {
            throw std::runtime_error("OpenCL - createBuffer(): the base already has a buffer!");
        }
        return buf;
    }

    // Retrieve a single buffer
    cl::Buffer *getBuffer(bh_base *base) {
        if (not util::exist(buffers, base)) {
            copyToDevice({base});
        }
        return buffers.at(base);
    }

    // Get C buffer from wrapped C++ object
    cl_mem getCBuffer(bh_base *base) {
        return (*getBuffer(base))();
    }

    // Get C context from wrapped C++ object
    cl_context getCContext() {
        return context();
    }

    // Get the OpenCL command queue object
    cl::CommandQueue *getQueue() {
        return &queue;
    }

    // Get C command queue from wrapped C++ object
    cl_command_queue getCQueue() {
        return queue();
    }

    // Update statistics with final aggregated values of the engine
    void updateFinalStatistics() override {
        stat.malloc_cache_lookups = malloc_cache.getTotalNumLookups();
        stat.malloc_cache_misses = malloc_cache.getTotalNumMisses();
    }
};

} // bohrium

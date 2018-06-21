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
#include <cuda.h>

namespace {
    // This will output the proper CUDA error strings
    // in the event that a CUDA host call returns an error
    #define check_cuda_errors(err) __check_cuda_errors(err, __FILE__, __LINE__)

    inline void __check_cuda_errors(CUresult err, const char *file, const int line) {
        if(CUDA_SUCCESS != err) {
            const char* err_name;
            cuGetErrorName(err, &err_name);
            const char* err_desc;
            cuGetErrorString(err, &err_desc);

            fprintf(stderr, "CUDA Error: %s \"%s\" from file <%s>, line %i.\n", err_name, err_desc, file, line);
            throw std::runtime_error("CUDA API call fail");
        }
    }
}

namespace bohrium {

class EngineCUDA : public jitk::EngineGPU {
private:
    // Map of all compiled OpenCL programs
    std::map<uint64_t, CUfunction> _functions;
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
    // A map of allocated buffers on the device
    std::map<bh_base*, CUdeviceptr> buffers;

    // The compiler to use when function doesn't exist
    jitk::Compiler compiler;

    // The hash of the JIT compilation command
    uint64_t compilation_hash;

    // Returns the block and thread sizes based on the 'threaded_blocks'
    std::pair<std::tuple<uint32_t, uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t>>
        NDRanges(const std::vector<uint64_t> &thread_stack) const;

    // Return a kernel function based on the given 'source'
    CUfunction getFunction(const std::string &source, const std::string &func_name);

    // We initialize the malloc cache with an alloc and free function
    MallocCache::FuncAllocT func_alloc = [](uint64_t nbytes) -> void * {
        CUdeviceptr new_buf;
        check_cuda_errors(cuMemAlloc(&new_buf, nbytes));
        return (void*) new_buf;
    };
    MallocCache::FuncFreeT func_free = [](void *mem, uint64_t nbytes) {
        check_cuda_errors(cuMemFree((CUdeviceptr) mem));
    };
    MallocCache malloc_cache{func_alloc, func_free, 0};

public:
    EngineCUDA(component::ComponentVE &comp, jitk::Statistics &stat);
    ~EngineCUDA() override;

    // Execute the 'source'
    void execute(const jitk::SymbolTable &symbols,
                 const std::string &source,
                 uint64_t codegen_hash,
                 const std::vector<uint64_t> &thread_stack,
                 const std::vector<const bh_instruction*> &constants);

    void writeKernel(const jitk::LoopB &kernel,
                     const jitk::SymbolTable &symbols,
                     const std::vector<uint64_t> &thread_stack,
                     uint64_t codegen_hash,
                     std::stringstream &ss) override;

    // Delete a buffer
    void delBuffer(bh_base* base) override {
        auto it = buffers.find(base);
        if (it != buffers.end()) {
            malloc_cache.free(base->nbytes(), (void*) it->second);
            buffers.erase(it);
        }
    }

    // Retrieve a single buffer
    template <typename T>
    CUdeviceptr *getBuffer(T &base) {
        if(buffers.find(base) == buffers.end()) {
            copyToDevice({ base });
        }
        return &buffers[base];
    }

    // Copy 'bases' to the host (ignoring bases that isn't on the device)
    void copyToHost(const std::set<bh_base*> &bases) override {
        auto tcopy = std::chrono::steady_clock::now();
        // Let's copy sync'ed arrays back to the host
        for(bh_base *base: bases) {
            if (buffers.find(base) != buffers.end()) {
                bh_data_malloc(base);
                if (verbose) {
                    std::cout << "Copy to host: " << *base << std::endl;
                }
                check_cuda_errors(cuMemcpyDtoH(base->data, buffers.at(base), base->nbytes()));
                // When syncing we assume that the host writes to the data and invalidate the device data thus
                // we have to remove its data buffer
                delBuffer(base);
            }
        }
        stat.time_copy2host += std::chrono::steady_clock::now() - tcopy;
    }

    // Copy 'base_list' to the device (ignoring bases that is already on the device)
    void copyToDevice(const std::set<bh_base*> &base_list) override {

        // Let's update the maximum memory usage on the device
        if (prof) {
            uint64_t sum = 0;
            for (const auto &b: buffers) {
                sum += b.first->nbytes();
            }
            stat.max_memory_usage = sum > stat.max_memory_usage ? sum : stat.max_memory_usage;
        }

        auto tcopy = std::chrono::steady_clock::now();
        for(bh_base *base: base_list) {
            if (buffers.find(base) == buffers.end()) { // We shouldn't overwrite existing buffers
                auto new_buf = reinterpret_cast<CUdeviceptr>(malloc_cache.alloc(base->nbytes()));
                buffers[base] = new_buf;

                // If the host data is non-null we should copy it to the device
                if (base->data != nullptr) {
                    if (verbose) {
                        std::cout << "Copy to device: " << *base << std::endl;
                    }
                    check_cuda_errors(cuMemcpyHtoD(new_buf, base->data, base->nbytes()));
                }
            }
        }
        stat.time_copy2dev += std::chrono::steady_clock::now() - tcopy;
    }

    // Copy all bases to the host (ignoring bases that isn't on the device)
    void copyAllBasesToHost() override {
        std::set<bh_base*> bases_on_device;
        for(auto &buf_pair: buffers) {
            bases_on_device.insert(buf_pair.first);
        }
        copyToHost(bases_on_device);
    }

    // Tell the engine to use the current CUDA context
    void useCurrentContext() {
        CUcontext new_context;
        check_cuda_errors(cuCtxGetCurrent(&new_context));
        if (new_context == nullptr or new_context == context) {
            return; // Nothing to do
        }

        // Empty the context for buffers and deallocate it
        copyAllBasesToHost();
        cuCtxDetach(context);

        // Save and attach (increase the refcount) the new context
        context = new_context;
        cuCtxAttach(&context, 0);

        // We have to clean all kernels compiled with the old context
        // Notice, the removed kernels are leaked when useCurrentContext()
        // isn't called as the first think (not really a big deal)
        _functions.clear();
    }

    // Sets the constructor flag of each instruction in 'instr_list'
    void setConstructorFlag(std::vector<bh_instruction*> &instr_list) override;

    // Return a YAML string describing this component
    std::string info() const;

    // Writes the CUDA specific for-loop header
    void loopHeadWriter(const jitk::SymbolTable &symbols,
                        jitk::Scope &scope,
                        const jitk::LoopB &block,
                        const std::vector<uint64_t> &thread_stack,
                        std::stringstream &out) override {
        // Write the for-loop header
        std::string itername;
        { std::stringstream t; t << "i" << block.rank; itername = t.str(); }
        if (thread_stack.size() > static_cast<uint64_t >(block.rank)) {
            assert(block._sweeps.size() == 0);
            out << "{ // Threaded block (ID " << itername << ")";
        } else {
            out << "for(" << writeType(bh_type::INT64) << " " << itername << " = 0; ";
            out << itername << " < " << block.size << "; ++" << itername << ") {";
        }
        out << "\n";
    }

    // Return CUDA types, which are used inside the JIT kernels
    const std::string writeType(bh_type dtype) override {
        switch (dtype) {
            case bh_type::BOOL:       return "bool";
            case bh_type::INT8:       return "char";
            case bh_type::INT16:      return "short";
            case bh_type::INT32:      return "int";
            case bh_type::INT64:      return "long";
            case bh_type::UINT8:      return "unsigned char";
            case bh_type::UINT16:     return "unsigned short";
            case bh_type::UINT32:     return "unsigned int";
            case bh_type::UINT64:     return "unsigned long";
            case bh_type::FLOAT32:    return "float";
            case bh_type::FLOAT64:    return "double";
            case bh_type::COMPLEX64:  return "cuFloatComplex";
            case bh_type::COMPLEX128: return "cuDoubleComplex";
            case bh_type::R123:       return "ulong2";
            default:
                std::cerr << "Unknown CUDA type: " << bh_type_text(dtype) << std::endl;
                throw std::runtime_error("Unknown CUDA type");
        }
    }

    const char *writeThreadId(unsigned int dim) {
        switch (dim) {
            case 0:
                return "(blockIdx.x * blockDim.x + threadIdx.x)";
            case 1:
                return "(blockIdx.y * blockDim.y + threadIdx.y)";
            case 2:
                return "(blockIdx.z * blockDim.z + threadIdx.z)";
            default:
                throw std::runtime_error("CUDA only support 3 dimensions");
        }
    }

    // Update statistics with final aggregated values of the engine
    void updateFinalStatistics() override {
        stat.malloc_cache_lookups = malloc_cache.getTotalNumLookups();
        stat.malloc_cache_misses = malloc_cache.getTotalNumMisses();
    }
};

} // bohrium

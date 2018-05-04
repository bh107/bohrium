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

#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <boost/filesystem.hpp>

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/block.hpp>
#include <jitk/compiler.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/codegen_cache.hpp>

#include <jitk/engines/engine_cpu.hpp>

namespace bohrium {

typedef void (*KernelFunction)(void* data_list[], uint64_t offset_strides[], bh_constant_value constants[]);

class EngineOpenMP : public jitk::EngineCPU {
private:
    std::map<uint64_t, KernelFunction> _functions;
    std::vector<void*> _lib_handles;

    // The compiler to use when function doesn't exist
    const jitk::Compiler compiler;

    // Return a kernel function based on the given 'source' and the name of the kernel function
    KernelFunction getFunction(const std::string &source, const std::string &func_name);

public:
    EngineOpenMP(component::ComponentVE &comp, jitk::Statistics &stat);

    ~EngineOpenMP() override;

    void execute(const jitk::SymbolTable &symbols,
                 const std::string &source,
                 uint64_t codegen_hash,
                 const std::vector<const bh_instruction*> &constants) override;

    void writeKernel(const jitk::LoopB &kernel,
                     const jitk::SymbolTable &symbols,
                     const std::vector<bh_base *> &kernel_temps,
                     uint64_t codegen_hash,
                     std::stringstream &ss) override;

     // Writing the OpenMP header, which include "parallel for" and "simd"
    void writeHeader(const jitk::SymbolTable &symbols,
                     jitk::Scope &scope,
                     const jitk::LoopB &block,
                     std::stringstream &out);

    void loopHeadWriter(const jitk::SymbolTable &symbols,
                        jitk::Scope &scope,
                        const jitk::LoopB &block,
                        const std::vector<uint64_t> &thread_stack,
                        std::stringstream &out) override;

    // Return a YAML string describing this component
    std::string info() const override;

    // Return C99 types, which are used inside the C99 kernels
    const std::string writeType(bh_type dtype) override;

    // Update statistics with final aggregated values of the engine
    void updateFinalStatistics() override {
        bh_get_malloc_cache_stat(stat.malloc_cache_lookups, stat.malloc_cache_misses, stat.max_memory_usage);
    }

private:
    // Writes the union of C99 types that can make up a constant
    inline void writeUnionType(std::stringstream& out) {
        out << "\ntypedef struct { uint64_t x, y; } r123_t" << ";\n";
        out << "union dtype {\n";
        util::spaces(out, 4); out << writeType(bh_type::BOOL)       << " " << bh_type_text(bh_type::BOOL)       << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::INT8)       << " " << bh_type_text(bh_type::INT8)       << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::INT16)      << " " << bh_type_text(bh_type::INT16)      << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::INT32)      << " " << bh_type_text(bh_type::INT32)      << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::INT64)      << " " << bh_type_text(bh_type::INT64)      << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::UINT8)      << " " << bh_type_text(bh_type::UINT8)      << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::UINT16)     << " " << bh_type_text(bh_type::UINT16)     << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::UINT32)     << " " << bh_type_text(bh_type::UINT32)     << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::UINT64)     << " " << bh_type_text(bh_type::UINT64)     << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::FLOAT32)    << " " << bh_type_text(bh_type::FLOAT32)    << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::FLOAT64)    << " " << bh_type_text(bh_type::FLOAT64)    << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::COMPLEX64)  << " " << bh_type_text(bh_type::COMPLEX64)  << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::COMPLEX128) << " " << bh_type_text(bh_type::COMPLEX128) << ";\n";
        util::spaces(out, 4); out << writeType(bh_type::R123)       << " " << bh_type_text(bh_type::R123)       << ";\n";
        out << "};\n";
    }
};
} // bohrium

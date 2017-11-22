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

#include <jitk/engine.hpp>

namespace bohrium {

typedef void (*KernelFunction)(void* data_list[], uint64_t offset_strides[], bh_constant_value constants[]);

class EngineOpenMP : public jitk::Engine {
  private:
    std::map<uint64_t, KernelFunction> _functions;
    std::vector<void*> _lib_handles;

    // The compiler to use when function doesn't exist
    const jitk::Compiler compiler;

    // Return a kernel function based on the given 'source'
    KernelFunction getFunction(const std::string &source);

  public:
    EngineOpenMP(const ConfigParser &config, jitk::Statistics &stat);

    ~EngineOpenMP();

    void execute(const std::string &source,
                 const std::vector<bh_base*> &non_temps,
                 const std::vector<const bh_view*> &offset_strides,
                 const std::vector<const bh_instruction*> &constants) override;

    void set_constructor_flag(std::vector<bh_instruction*> &instr_list) override;

    void handle_execution(BhIR *bhir) override;

    void write_kernel(const std::vector<jitk::Block> &block_list,
                      const jitk::SymbolTable &symbols,
                      const std::vector<bh_base*> &kernel_temps,
                      std::stringstream &ss) override;

     // Writing the OpenMP header, which include "parallel for" and "simd"
    void write_header(const jitk::SymbolTable &symbols,
                      jitk::Scope &scope,
                      const jitk::LoopB &block,
                      std::stringstream &out);

    void loop_head_writer(const jitk::SymbolTable &symbols,
                          jitk::Scope &scope,
                          const jitk::LoopB &block,
                          bool loop_is_peeled,
                          const std::vector<const jitk::LoopB *> &threaded_blocks,
                          std::stringstream &out) override;

    // Return a YAML string describing this component
    std::string info() const override;

    // Return C99 types, which are used inside the C99 kernels
    const std::string write_type(bh_type dtype) override {
        switch (dtype) {
            case bh_type::BOOL:       return "bool";
            case bh_type::INT8:       return "int8_t";
            case bh_type::INT16:      return "int16_t";
            case bh_type::INT32:      return "int32_t";
            case bh_type::INT64:      return "int64_t";
            case bh_type::UINT8:      return "uint8_t";
            case bh_type::UINT16:     return "uint16_t";
            case bh_type::UINT32:     return "uint32_t";
            case bh_type::UINT64:     return "uint64_t";
            case bh_type::FLOAT32:    return "float";
            case bh_type::FLOAT64:    return "double";
            case bh_type::COMPLEX64:  return "float complex";
            case bh_type::COMPLEX128: return "double complex";
            case bh_type::R123:       return "r123_t"; // Defined by `write_c99_dtype_union()`
            default:
                std::cerr << "Unknown C99 type: " << bh_type_text(dtype) << std::endl;
                throw std::runtime_error("Unknown C99 type");
        }
    }

  private:
    // Writes the union of C99 types that can make up a constant
    inline void write_union_type(std::stringstream& out) {
        out << "\ntypedef struct { uint64_t x, y; } r123_t" << ";\n";
        out << "union dtype {\n";
        util::spaces(out, 4); out << write_type(bh_type::BOOL)       << " " << bh_type_text(bh_type::BOOL)       << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::INT8)       << " " << bh_type_text(bh_type::INT8)       << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::INT16)      << " " << bh_type_text(bh_type::INT16)      << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::INT32)      << " " << bh_type_text(bh_type::INT32)      << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::INT64)      << " " << bh_type_text(bh_type::INT64)      << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::UINT8)      << " " << bh_type_text(bh_type::UINT8)      << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::UINT16)     << " " << bh_type_text(bh_type::UINT16)     << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::UINT32)     << " " << bh_type_text(bh_type::UINT32)     << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::UINT64)     << " " << bh_type_text(bh_type::UINT64)     << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::FLOAT32)    << " " << bh_type_text(bh_type::FLOAT32)    << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::FLOAT64)    << " " << bh_type_text(bh_type::FLOAT64)    << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::COMPLEX64)  << " " << bh_type_text(bh_type::COMPLEX64)  << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::COMPLEX128) << " " << bh_type_text(bh_type::COMPLEX128) << ";\n";
        util::spaces(out, 4); out << write_type(bh_type::R123)       << " " << bh_type_text(bh_type::R123)       << ";\n";
        out << "};\n";
    }
};
} // bohrium

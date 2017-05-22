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

#ifndef __BH_VE_UNI_STORE_HPP
#define __BH_VE_UNI_STORE_HPP

#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <boost/filesystem.hpp>

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/kernel.hpp>
#include <jitk/block.hpp>

#include "compiler.hpp"

namespace bohrium {

typedef void (*KernelFunction)(void* data_list[], uint64_t offset_strides[], bh_constant_value constants[]);

class EngineOpenMP {
  private:
    std::map<uint64_t, KernelFunction> _functions;
    std::vector<void*> _lib_handles;

    // Path to a temporary directory for the source and object files
    const boost::filesystem::path tmp_dir;

    // Path to the directory of the source files
    const boost::filesystem::path source_dir;

    // Path to the directory of the object files
    const boost::filesystem::path object_dir;

    // The compiler to use when function doesn't exist
    const Compiler compiler;

    // Verbose flag
    const bool verbose;

    // Some statistics
    jitk::Statistics &stat;

    // Return a kernel function based on the given 'source'
    KernelFunction getFunction(const std::string &source);

  public:
    EngineOpenMP(const ConfigParser &config, jitk::Statistics &stat);
    ~EngineOpenMP();

    // The following methods implements the methods required by jitk::handle_execution()

    void execute(const std::string &source, const jitk::Kernel &kernel,
                 const std::vector<const jitk::LoopB*> &threaded_blocks,
                 const std::vector<const bh_view*> &offset_strides,
                 const std::vector<const bh_instruction*> &constants);
    void set_constructor_flag(std::vector<bh_instruction*> &instr_list);
    // Notice, OpenMP has no device thus the device methods does nothing
    template <typename T>
    void copyToHost(T &bases) {}
    template <typename T>
    void copyToDevice(T &base_list) {}
    template <typename T>
    void delBuffer(T &base) {}
};



} // bohrium

#endif

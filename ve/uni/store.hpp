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
#include "compiler.hpp"

namespace bohrium {

typedef void (*KernelFunction)(void* data_list[]);

class Store {
  private:
    std::map<uint64_t, KernelFunction> _functions;
    std::vector<void*> _lib_handles;

    // Path to the directory of the source files
    const boost::filesystem::path source_dir;

    // Path to the directory of the object files
    const boost::filesystem::path object_dir;

    // The compiler to use when function doesn't exist
    const Compiler compiler;

    // Whether we should write the kernel sources (.c files)
    const bool dump_src;

  public:
    Store(const ConfigParser &config);
    ~Store();

    KernelFunction getFunction(const std::string &source);
};



} // bohrium

#endif

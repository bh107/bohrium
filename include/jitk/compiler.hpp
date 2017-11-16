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

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <bh_config_parser.hpp>

namespace bohrium {
namespace jitk {

/**
 * compile() forks and executes a system process.
 */
class Compiler {
public:
    std::string cmd_template;
    std::string config_path;
    bool verbose;

    Compiler(std::string cmd_template, bool verbose, std::string config_path);
    Compiler() = default;

    /**
     *  Compile by piping, the given sourcecode into a shared object.
     *
     *  Throws runtime_error on compilation failure
     */
    void compile(std::string object_abspath, const char* sourcecode,size_t source_len) const;

    /**
     *  Compile source on disk.
     */

    /**
     *  Compile by disk writing, the given sourcecode into a shared object.
     *
     *  Throws runtime_error on compilation failure
     */
    void compile(std::string object_abspath, std::string src_abspath) const;
};

}}

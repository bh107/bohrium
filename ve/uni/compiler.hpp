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

#ifndef __BH_VE_UNI_COMPILER_HPP
#define __BH_VE_UNI_COMPILER_HPP

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>

namespace bohrium{

class Compiler {
public:
	/**
	 * compile() forks and executes a system process, the process along with
	 * arguments must be provided as argument at time of construction.
	 * The process must be able to consume sourcecode via stdin and produce
	 * a shared object file.
	 * The compiled shared-object is then loaded and made available for execute().
	 *
	 * Examples:
	 *
	 *  Compiler tcc("tcc", "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *  Compiler icc("ic",  "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *  Compiler gcc("gcc", "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *
	 */
    Compiler(std::string cmd, std::string inc, std::string lib, std::string flg, std::string ext);

    std::string text() const;
    std::string process_str(std::string object_abspath, std::string source_abspath) const;

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
     *  Compile by disk writting, the given sourcecode into a shared object.
     *
     *  Throws runtime_error on compilation failure
     */
    void compile(std::string object_abspath, std::string src_abspath) const;

private:
    std::string cmd_, inc_, lib_, flg_, ext_;
};

}

#endif

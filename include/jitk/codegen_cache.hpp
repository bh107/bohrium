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
#include <string>

#include <bh_instruction.hpp>
#include <jitk/block.hpp>
#include <jitk/statistics.hpp>


namespace bohrium {
namespace jitk {

class CodegenCache {
private:
    std::map<size_t, std::string> _cache;
    // Some statistics
    jitk::Statistics &stat;
public:
    // The constructor takes the statistic object
    explicit CodegenCache(jitk::Statistics &stat) : stat(stat) {}

    /** Check the cache for a source code that matches `kernel`
     *
     * @param kernel  The kernel
     * @param symbols The symbol table
     * @return The source code and the hash of the source or the empty string on cache misses
     */
    std::pair<std::string, uint64_t> lookup(const LoopB &kernel, const SymbolTable &symbols);

    /** Insert `source` as a hit when requesting `kernel`
     *
     * @param source  The source code
     * @param kernel  The kernel
     * @param symbols The symbol table
     */
    void insert(std::string source, const LoopB &kernel, const SymbolTable &symbols);
};

} // jit
} // bohrium

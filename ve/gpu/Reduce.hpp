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

#ifndef __REDUCE_HPP
#define __REDUCE_HPP

#include <string>
#include <map>
#include <bh.h>
#include "Kernel.hpp"
#include "StringHasher.hpp"
#include "UserFuncArg.hpp"

namespace Reduce
{
    typedef std::map<size_t, Kernel> KernelMap;
    static KernelMap kernelMap;
    bh_error bh_reduce(bh_instruction* inst, UserFuncArg* userFuncArg);
    Kernel getKernel(bh_instruction* inst,
                     UserFuncArg* userFuncArg,
                     std::vector<bh_index> shape);
    std::string generateCode(bh_instruction* inst,
                             OCLtype outType, OCLtype inType,
                             std::vector<bh_index> shape);
}

#endif

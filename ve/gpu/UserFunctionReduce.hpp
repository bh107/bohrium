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

#ifndef __USERFUNCTIONREDUCE_HPP
#define __USERFUNCTIONREDUCE_HPP

#include <string>
#include <map>
#include <bh.h>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"

namespace UserFunctionReduce
{
    typedef std::map<size_t, Kernel> KernelMap;
    static KernelMap kernelMap;
    void reduce(bh_reduce_type* reduceDef, UserFuncArg* userFuncArg);
	bh_error reduce_impl(bh_userfunc* arg, void* ve_arg);
    Kernel getKernel(bh_reduce_type* reduceDef,
                     UserFuncArg* userFuncArg,
                     std::vector<bh_index> shape);
    std::string generateCode(bh_reduce_type* reduceDef,
                             OCLtype outType, OCLtype inType,
                             std::vector<bh_index> shape);
}

#endif

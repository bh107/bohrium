/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __USERFUNCTIONREDUCE_HPP
#define __USERFUNCTIONREDUCE_HPP

#include <string>
#include <map>
#include <cphvb.h>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"

namespace UserFunctionReduce
{
    typedef std::map<size_t, Kernel> KernelMap;
    static KernelMap kernelMap;
    void reduce(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg);
    Kernel getKernel(cphvb_reduce_type* reduceDef,
                     UserFuncArg* userFuncArg,
                     std::vector<cphvb_index> shape);
    std::string generateCode(cphvb_reduce_type* reduceDef,
                             OCLtype outType, OCLtype inType,
                             std::vector<cphvb_index> shape);
}

#endif

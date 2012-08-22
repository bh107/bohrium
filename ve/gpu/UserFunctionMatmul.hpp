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

#ifndef __USERFUNCTIONMATMUL_HPP
#define __USERFUNCTIONMATMUL_HPP

#include <string>
#include <map>
#include <cphvb.h>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"

namespace UserFunctionMatmul
{
    typedef std::map<size_t, Kernel> KernelMap;
    static KernelMap kernelMap;
    void matmul(cphvb_matmul_type* matmulDef, UserFuncArg* userFuncArg);
    Kernel getKernel(cphvb_matmul_type* matmulDef,
                     UserFuncArg* userFuncArg);
    std::string generateCode(cphvb_matmul_type* matmulDef,
                             OCLtype type);
    std::string generateDefines(cphvb_matmul_type* matmulDef,
                             OCLtype type);
}

#endif

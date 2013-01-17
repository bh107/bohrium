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

#ifndef __USERFUNCTIONRANDOM_HPP
#define __USERFUNCTIONRANDOM_HPP

#include <map>
#include <bh.h>
#include <CL/cl.hpp>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"

class UserFunctionRandom
{
private:
    typedef std::map<bh_type, Kernel> KernelMap;
    KernelMap kernelMap;
    ResourceManager* resourceManager;
    Buffer* state;
    static void CL_CALLBACK hostDataDelete(cl_event ev, cl_int eventStatus, void* data);
public:
    UserFunctionRandom(ResourceManager* rm);
    bh_error fill(UserFuncArg* userFuncArg);
};

#endif

/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __USERFUNCTIONRANDOM_HPP
#define __USERFUNCTIONRANDOM_HPP

#include <map>
#include <cphvb_random.h>
#include <CL/cl.hpp>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"

namespace UserFunctionRandom
{
    typedef std::map<cphvb_type, Kernel> KernelMap;
    static KernelMap kernelMap = KernelMap();

    static ResourceManager* resourceManager = NULL;
    static Buffer* state;

    void initialize();
    void finalize();
    void CL_CALLBACK hostDataDelete(cl_event ev, cl_int eventStatus, void* data);
    void run(UserFuncArg* userFuncArg);
}

#endif

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

#ifndef __USERFUNCTIONREDUCE_HPP
#define __USERFUNCTIONREDUCE_HPP

#include <string>
#include <map>
#include <cphvb_reduce.h>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"

namespace UserFunctionReduce
{
    typedef std::map<size_t, Kernel> KernelMap;
    static KernelMap kernelMap;
    static std::hash<std::string> strHash;
    void reduce(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg);
    Kernel getKernel(cphvb_reduce_type* reduceDef,
                     UserFuncArg* userFuncArg,
                     std::vector<cphvb_index> shape);
    std::string generateCode(cphvb_reduce_type* reduceDef,
                             OCLtype outType, OCLtype inType,
                             std::vector<cphvb_index> shape);
}

#endif

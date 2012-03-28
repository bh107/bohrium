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

#include <cphvb_random.h>


namespace UserFunctionRandom
{
    static ResourceManager* resourceManager = NULL;
    static BaseArray* state;
    static cphvb_array init_array;

    void initialize();
    void finalize();
    void CL_CALLBACK hostDataDelete(cl_event ev, cl_int eventStatus, void* data)
    void run(cphvb_reduce_type* reduceDef, UserFuncArg* userFuncArg);
    Kernel generateKernel(cphvb_reduce_type* reduceDef, 
                          UserFuncArg* userFuncArg,
                          const std::vector<cphvb_index>& shape);
    std::string generateCode(cphvb_reduce_type* reduceDef, 
                             const std::vector<BaseArray*>& operandBase,
                             const std::vector<cphvb_index>& shape);
}

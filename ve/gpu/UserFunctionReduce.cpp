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

#include <iostream>
#include <sstream>
#include <cphvb_reduce.h>

cphvb_error cphvb_reduce(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_reduce_type* reduceDef = (cphvb_reduce_type*)arg;
    std::vector<BaseArray*>* operandBase = (std::vector<BaseArray*>*)ve_arg;
    assert(reduceDef->nout = 1);
    assert(reduceDef->nin = 1);
    assert(reduceDef->operand[0]->ndim + 1 == reduceDef->operand[1]->ndim);
    assert(reduceDef->operand[0]->type == reduceDef->operand[1]->type);
    userFunctionReduce->run(reduceDef, operandBase);
}

std::pair<std::string, std::string> UserFunctionReduce::generateCode(cphvb_reduce_type* reduceDef, 
                                                                     const std::vector<BaseArray*>& operandBase)
{
    cphvb_array* out = reduceDef->operand[0];
    std::stringstream source;
    source << "__kernel void " << kernelName << "(" <<
        " __global " << oclTypeStr(operandBase[0]->type()) << "* out,"
        " __global " << oclTypeStr(operandBase[1]->type()) << "* in)\n{\n";
    
    if (out->ndim > 2)
        source << "\tconst size_t gidz = get_global_id(2);\n";
    if (out->ndim > 1)
        source << "\tconst size_t gidy = get_global_id(1);\n";
    source << "\tconst size_t gidx = get_global_id(0);\n";
    
    
}

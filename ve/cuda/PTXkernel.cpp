/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include "PTXkernel.hpp"


PTXkernel::PTXkernel(PTXversion version_,
                     CUDAtarget target_,
                     PTXregisterBank* registerBank_,
                     PTXinstructionList* instructionList_) :
    version(version_),
    target(target_),
    parameterCount(0),
    registerBank(registerBank_),
    instructionList(instructionList_) {}

void PTXkernel::clear()
{
    parameterCount = 0;
    //registerBank and instructioList is cleared form KernelGenerator
}

PTXkernelParameter* PTXkernel::addParameter(PTXtype type)
{
    parameterList[parameterCount].type = type;
    parameterList[parameterCount].id = parameterCount;
    return &parameterList[parameterCount++];
}

int PTXkernel::snprint(char* buf, int size)
{
    int res = 0;
    int bp;
    bp = std::snprintf(buf, size, ".version %s\n.target %s\n.entry %s (", 
                       ptxVersionStr(version),
                       cudaTargetStr(target),
                       name);
    res += bp; buf += bp; size -= bp;
    if (parameterCount > 0)
    {
        bp = parameterList[0].declare("\t",buf, size);
        res += bp; buf += bp; size -= bp;
    }
    for (int i = 1; i < parameterCount; ++i)
    {
        bp = parameterList[i].declare(",\n\t",buf, size);
        res += bp; buf += bp; size -= bp;
    }
    bp = std::snprintf(buf, size, ")\n{\n");
    res += bp; buf += bp; size -= bp;
    bp = registerBank->declare(buf,size);
    res += bp; buf += bp; size -= bp;
    bp = instructionList->snprint(buf,size);
    res += bp; buf += bp; size -= bp;
    bp = std::snprintf(buf, size, "}\n");
    return res + bp;
}



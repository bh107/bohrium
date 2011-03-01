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
                     PTXkernelBody* kernelBody_) :
    version(version_),
    target(target_),
    parameterCount(0),
    registerBank(registerBank_),
    kernelBody(kernelBody_) {}

PTXkernelParameter* PTXkernel::addParameter(PTXtype type)
{
    parameterList[parameterCount].type = type;
    parameterList[parameterCount].id = parameterCount;
    return &parameterList[parameterCount++];
}

const char* versionStr[] =  
{
    /*[ISA_14] = */"1.4",
    /*[ISA_22] = */"2.2"
};

const char* targetStr[] =  
{
    /*[SM_10] = */"sm_10",
    /*[SM_11] = */"sm_11",
    /*[SM_12] = */"sm_12",
    /*[SM_13] = */"sm_13",
    /*[SM_20] = */"sm_20"
};

int PTXkernel::snprint(char* buf, int size)
{
    int res = 0;
    int bp;
    bp = std::snprintf(buf, size, ".version %s\n.target %s\n.entry %s (", 
                       versionStr[version],
                       targetStr[target],
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
    bp = kernelBody->snprint(buf,size);
    res += bp; buf += bp; size -= bp;
    bp = std::snprintf(buf, size, "}\n");
    return res + bp;
}



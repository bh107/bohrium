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

#include <cassert>
#include <cstdio>
#include <cphvb.h>
#include "PTXoperand.hpp"
#include "PTXregister.hpp"
#include "PTXregisterBank.hpp"

PTXregisterBank::PTXregisterBank() 
{
    reset();
}

void PTXregisterBank::reset()
{
    next = 0;
    int i;
    for (i = 0; i < PTX_TYPES; ++i)
    {
        instanceTable[i] = 0;
    }
}

PTXregister* PTXregisterBank::newRegister(PTXtype type)
{
    registers[next].type = type;
    registers[next].typeIdx = instanceTable[type]++;
    return &registers[next++];
}


PTXregister* PTXregisterBank::newRegister(cphvb_type vbtype)
{
    return newRegister(ptxType(vbtype));
}

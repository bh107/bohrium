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

PTXregisterBank::PTXregisterBank() {}

void PTXregisterBank::reset()
{
    next = 0;
    int i,j;
    for (i = 0; i < PTX_TYPES; ++i)
    {
        for (j = 0; j < PTX_SIZES; ++j)
        {
            instanceTable[i][j] = 0;
        }
    }
}
 
const char* regTypeString(PTXtype type)
{
    switch (type)
    {
    case PTX_INT:
        return "i";
    case PTX_UINT:
        return "u";
    case PTX_FLOAT:
        return "f";
    case PTX_BITS:
        return "b";
    case PTX_PRED:
        return "p";
    default:
        assert(false);
    }
}

const char* regSizeString(RegSize size)
{
    switch (size)
    {
    case _8:
        return "c";
    case _16:
        return "h";
    case _32:
        return "_";
    case _64:
        return "d";
    default:
        assert(false);
    }
}

void PTXregisterBank::setRegName(PTXtype type, RegSize size, char* name)
{
    std::sprintf(name, "$%s%s%d", 
                 regTypeString(type),
                 regSizeString(size),
                 instanceTable[type][size]);
}

PTXregister* PTXregisterBank::newRegister(PTXtype type, RegSize size)
{
    registers[next].type = type;
    registers[next].size = size;
    registers[next].name = names[next];
    setRegName(type, size, names[next]);
    ++instanceTable[type][size];
    return &registers[next++];
}


PTXregister* PTXregisterBank::newRegister(cphvb_type vbtype)
{
    PTXtype type = PTXoperand::ptxType(vbtype);
    RegSize size;
    switch (cphvb_type_size(vbtype))
    {
    case 1:
        size = _8;
        break;
    case 2:
        size = _16;
        break;
    case 4:
        size = _32;
        break;
    case 8:
        size = _64;
        break;
    default:
        assert(false);
    }
    return newRegister(type,size);
}

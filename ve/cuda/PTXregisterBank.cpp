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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cphvb.h>
#include "PTXoperand.hpp"
#include "PTXregister.hpp"
#include "PTXspecialRegister.hpp"
#include "PTXregisterBank.hpp"

PTXregisterBank::PTXregisterBank() :
    registers(new StaticStack<PTXregister>(1024)),
    tid_x(PTXspecialRegister(TID_X)),
    ntid_x(PTXspecialRegister(NTID_X)),
    ctaid_x(PTXspecialRegister(CTAID_X))
{
    for (int i = 0; i < PTX_TYPES; ++i)
    {
        instanceTable[i] = 0;
    }
}

void PTXregisterBank::clear()
{
    registers->clear();
    for (int i = 0; i < PTX_TYPES; ++i)
    {
        instanceTable[i] = 0;
    }
}

PTXregister* PTXregisterBank::next(PTXtype type)
{
    return registers->next(type,instanceTable[type]++);
}

inline void PTXregisterBank::declareOn(std::ostream& os) const
{
    for (int i = 0; i < PTX_TYPES; ++i)
    {
        if (instanceTable[i] > 0)
        {
            os << std::setw(10) << "";
            os << ".reg " << ptxTypeStr((PTXtype)i) << " " << 
                ptxRegPrefix((PTXtype)i) << " <" << instanceTable[i] << ">;\n";
        }
    }
}

std::ostream& operator<<= (std::ostream& os, 
                                  PTXregisterBank const& ptxRegisterBank)
{
    ptxRegisterBank.declareOn(os);
    return os;
}

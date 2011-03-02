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

#ifndef __PTXINSTRUCTION_HPP
#define __PTXINSTRUCTION_HPP

#include <string>
#include "PTXopcode.h"
#include "PTXtype.h"
#include "PTXoperand.hpp"
#include "PTXregister.hpp"

class PTXinstruction
{
    friend class PTXinstructionList;
private:
    char* label;
    bool guardMod;   //guard modifier 
    PTXregister* guard; //the guard predicate register. NULL if not used.
    PTXopcode opcode;
    PTXregister* dest;
    PTXoperand* src[PTX_MAX_OPERANDS-1];
    int snprintAritOp(char* buf, int size);
    int snprintOp(char* buf, int size);
public:
    int snprint(char* buf, int size);
};

#endif

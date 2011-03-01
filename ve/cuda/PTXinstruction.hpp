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
#include "PTXoperand.hpp"
#include "PTXregister.hpp"

enum Opcode
{
    PTX_BRA,
    PTX_ADD,
    PTX_SUB,
    PTX_MUL,
    PTX_MAD,
    PTX_DIV,
    PTX_SAD,
    PTX_REM,
    PTX_STORE,
    PTX_LOAD,
    PTX_LDPARAM,
    PTX_MOV
};

#define MAX_OPERANDS (4)

class PTXinstruction
{
private:
    char* label;
    bool guardMod;   //guard modifier 
    PTXregister* guard; //the guard predicate register. NULL if not used.
    Opcode opcode;
    PTXregister* dest;
    PTXoperand* src[MAX_OPERANDS-1];
    int snprintAritOp(char* buf, int size);
    int snprintOp(char* buf, int size);
public:
    int snprint(char* buf, int size);
};


#endif

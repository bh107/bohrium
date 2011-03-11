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
#include "PTXinstructionList.hpp"

PTXinstructionList::PTXinstructionList() :
    next(0) {}

void PTXinstructionList::clear()
{
    next = 0;
}

void PTXinstructionList::add(char* label,
                             bool guardMod,
                             PTXregister* guard,
                             PTXopcode opcode,
                             PTXregister* dest,
                             PTXoperand* src[])
{
    instructions[next].label = label;
    instructions[next].guardMod = guardMod;
    instructions[next].guard = guard;
    instructions[next].opcode = opcode;
    instructions[next].dest = dest;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        instructions[next].src[i] = src[i];
    }
    ++next;
}

void PTXinstructionList::add(PTXopcode opcode,
                             PTXregister* dest,
                             PTXoperand* src[])
{
    add(NULL,false,NULL,opcode,dest,src);
}

void PTXinstructionList::add(PTXopcode opcode,
                             PTXregister* dest)
{
    add(NULL,false,NULL,opcode,dest,NULL);
}

void PTXinstructionList::add(PTXopcode opcode,
                             PTXregister* dest,
                             PTXoperand* src1)
{
    add(NULL,false,NULL,opcode,dest,&src1);
}

void PTXinstructionList::add(PTXopcode opcode,
                             PTXregister* dest,
                             PTXoperand* src1,             
                             PTXoperand* src2)
{
    add(NULL,false,NULL,opcode,dest,(PTXoperand*[]){src1,src2});
}

void PTXinstructionList::add(PTXopcode opcode,
                             PTXregister* dest,
                             PTXoperand* src1,
                             PTXoperand* src2,             
                             PTXoperand* src3)
{
    add(NULL,false,NULL,opcode,dest,(PTXoperand*[]){src1,src2,src3});
}
void PTXinstructionList::add(PTXregister* guard,
                             PTXopcode opcode,
                             char* label)
{
    assert(opcode == PTX_BRA);
    add(label,true,guard,opcode,NULL,NULL);
}

void PTXinstructionList::add(PTXregister* guard,
                             PTXopcode opcode)
{
    add(NULL,true,guard,opcode,NULL,NULL);
}

void PTXinstructionList::add(PTXopcode opcode)
{
    assert(opcode == PTX_EXIT);
    add(NULL,true,NULL,opcode,NULL,NULL);
}

int PTXinstructionList::snprint(char* buf, 
                                int size)
{
    int res = 0;
    int bp;
    for (int i = 0; i < next; ++i)
    {
        bp = instructions[i].snprint(buf,size);
        res += bp; buf += bp; size -= bp;
    }
    return res;
}

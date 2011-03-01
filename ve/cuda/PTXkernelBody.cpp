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

#include "PTXkernelBody.hpp"

PTXkernelBody::PTXkernelBody() :
    next(0) {}

void PTXkernelBody::reset()
{
    next = 0;
}

void PTXkernelBody::add(char* label,
                        bool guardMod,
                        PTXregister* guard,
                        Opcode opcode,
                        PTXregister* dest,
                        PTXoperand* src[])
{
    instructions[next].label = label;
    instructions[next].guardMod = guardMod;
    instructions[next].guard = guard;
    instructions[next].opcode = opcode;
    instructions[next].dest = dest;
    for (int i = 0; i < PTXinstruction::opSrc(opcode); ++i)
    {
        instructions[next].src[i] = src[i];
    }
}

void PTXkernelBody::add(Opcode opcode,
                      PTXregister* dest,
                      PTXoperand* src[])
{
    add(NULL,false,NULL,opcode,dest,src);
}

int PTXkernelBody::snprint(char* buf, int size)
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

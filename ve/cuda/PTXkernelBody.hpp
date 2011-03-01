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

#ifndef __PTXKERNELBODY
#define __PTXKERNELBODY

#include "PTXinstruction.hpp"
#include "PTXregister.hpp"
#include "PTXoperand.hpp"


#define BUFFERSIZE (4096)

class PTXkernelBody
{
private:
    PTXinstruction instructions[BUFFERSIZE];
    int next;
public:
    PTXkernelBody();
    void reset();
    void add(char* label,
             bool guardMod,
             PTXregister* guard,
             Opcode opcode,
             PTXregister* dest,
             PTXoperand* src[]);
    void add(Opcode opcode,
             PTXregister* dest,
             PTXoperand* src[]);
    int snprint(char* buf, int size);
};

#endif

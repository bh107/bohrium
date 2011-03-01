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
#include <stdexcept>
#include "PTXinstruction.hpp"

const int _opSrc[] =
{
    /*PTX_BRA = */0, 
    /*PTX_ADD = */2,
    /*PTX_SUB = */2,
    /*PTX_MUL = */2,
    /*PTX_MAD = */3,
    /*PTX_DIV = */2,
    /*PTX_SAD = */3,
    /*PTX_REM = */2,
    /*PTX_STORE = */2,
    /*PTX_LOAD = */2,
    /*PTX_LDPARAM = */1,
    /*PTX_MOV = */1
};

const int PTXinstruction::opSrc(Opcode opcode)
{
    return _opSrc[opcode];
}

int PTXinstruction::snprintAritOp(char* buf, int size)
{ 
    int res = 0;
    int bp;
    switch (opcode)
    {
    case PTX_ADD:
        bp = std::snprintf(buf, size, "add.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_SUB:
        bp = std::snprintf(buf, size, "sub.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MUL:
        bp = std::snprintf(buf, size, "mul.lo.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MAD:
        bp = std::snprintf(buf, size, "mad.lo.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_DIV:
        bp = std::snprintf(buf, size, "div.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_REM:
        bp = std::snprintf(buf, size, "rem.%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MOV:
        bp = std::snprintf(buf, size, "mov.%s\t", ptxTypeStr(dest->type));
        break;
    default:
        assert (false);
     }
    res += bp; buf += bp; size -= bp;
    dest->snprint(buf,size);
    for (int i = 0; i < opSrc(opcode); ++i)
    {
        bp = src[i]->snprint(", ",buf,size); 
        res += bp; buf += bp; size -= bp;
    }
    bp = std::snprintf(buf, size,";\n");
    return res + bp;
}

int PTXinstruction::snprintOp(char* buf, int size)
{
    int res = 0;
    int bp;
    switch (opcode)
    {
    case PTX_BRA:
        return std::snprintf(buf, size, "bra\t%s;\n", label);
    case PTX_LOAD:
        bp = std::snprintf(buf, size, "ld.global.%s\t", 
                           ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(buf,size,", ");
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"+");
        res += bp; buf += bp; size -= bp;
        bp = src[1]->snprint(buf,size,"];\n");
        return res + bp;
    case PTX_LDPARAM:
        bp = std::snprintf(buf, size, "ld.param.%s\t", ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(buf,size,", ");
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"];\n");
        return res + bp;
    case PTX_STORE:
        bp = std::snprintf(buf, size, "st.global.%s\t", 
                           ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"+");
        res += bp; buf += bp; size -= bp;
        bp = src[1]->snprint(buf,size,"]");
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(", ",buf,size,";\n");
        return res + bp;
    default:
        return snprintAritOp(buf, size);
    }
}

int PTXinstruction::snprint(char* buf, int size)
{
    int res = 0;
    int bp;
    if (label != NULL && opcode != PTX_BRA)    
    {
        bp = snprintf(buf, size, "%s:\n",label);
        res += bp; buf += bp; size -= bp;
    }
    if (guard != NULL)
    {
        bp = snprintf(buf, size, "@%s",guardMod?"":"!");
        res += bp; buf += bp; size -= bp;
        bp = guard->snprint(buf,size);
        res += bp; buf += bp; size -= bp;
    }
    bp = snprintOp(buf, size);
    res += bp;
    if (bp > size)
    {
        throw std::runtime_error("Not enough buffer space for printing.");
    }
    return res;
}



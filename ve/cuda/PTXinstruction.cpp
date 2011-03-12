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
#include <cstdio>
#include <stdexcept>
#include "PTXinstruction.hpp"

int PTXinstruction::snprintAritOp(char* buf, int size)
{ 
    int res = 0;
    int bp;
    switch (opcode)
    {
    case PTX_ADD:
        bp = std::snprintf(buf, size, "\tadd%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_SUB:
        bp = std::snprintf(buf, size, "\tsub%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MUL:
        bp = std::snprintf(buf, size, "\tmul%s%s\t", 
                           (ptxBaseType(dest->type) == PTX_FLOAT)?"":".lo", 
                           ptxTypeStr(dest->type));
        break;
    case PTX_MAD:
        bp = std::snprintf(buf, size, "\tmad.lo%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MAD_WIDE:
        bp = std::snprintf(buf, size, "\tmad.wide%s\t",
                           ptxWideOpStr(dest->type));
        break;
    case PTX_DIV:
        bp = std::snprintf(buf, size, "\tdiv%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_REM:
        bp = std::snprintf(buf, size, "\trem%s\t", ptxTypeStr(dest->type));
        break;
    case PTX_MOV:
        bp = std::snprintf(buf, size, "\tmov%s\t", ptxTypeStr(dest->type));
        break;
    default:
        assert (false);
     }
    res += bp; buf += bp; size -= bp;
    bp = dest->snprint(buf,size);
    res += bp; buf += bp; size -= bp;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        bp = src[i]->snprint(", ",buf,size); 
        res += bp; buf += bp; size -= bp;
    }
    bp = std::snprintf(buf, size,";\n");
    return res + bp;
}

int PTXinstruction::snprintLogicOp(char* buf, int size)
{ 
    int res = 0;
    int bp;
    PTXregister* srcReg;
    PTXtype type = PTX_INT32;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        srcReg = dynamic_cast<PTXregister*>(src[i]);
        if (srcReg != NULL)
        {
            type = srcReg->type;
            break;
        }
    }
    switch (opcode)
    {    
    case PTX_SETP_GE:
        bp = std::snprintf(buf, size, "\tsetp.ge%s\t", ptxTypeStr(type));
        break;
    default:
        assert (false);
    }
    res += bp; buf += bp; size -= bp;
    bp = dest->snprint(buf,size);
    res += bp; buf += bp; size -= bp;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        bp = src[i]->snprint(", ",buf,size); 
        res += bp; buf += bp; size -= bp;
    }
    bp = std::snprintf(buf, size,";\n");
    return res + bp;    
}

int PTXinstruction::snprintConvertOp(char* buf, int size)
{
    int res = 0;
    int bp;
    PTXregister* srcReg = dynamic_cast<PTXregister*>(src[0]);
    assert (srcReg != NULL);
    bp = std::snprintf(buf, size, "\tcvt.rn%s%s\t", ptxTypeStr(dest->type),
                         ptxTypeStr(srcReg->type));
    res += bp; buf += bp; size -= bp;
    bp = dest->snprint(buf,size);
    res += bp; buf += bp; size -= bp;
    bp = srcReg->snprint(", ",buf,size,";\n");
    return res + bp;    
}

int PTXinstruction::snprintOp(char* buf, int size)
{
    int res = 0;
    int bp;
    switch (opcode)
    {
    case PTX_EXIT:
        return std::snprintf(buf, size, "\texit;\n");
    case PTX_MEMBAR:
        return std::snprintf(buf, size, "\tmembar.gl;\n");
    case PTX_BRA:
        return std::snprintf(buf, size, "\tbra\t%s;\n", label);
    case PTX_LD_GLOBAL:
        bp = std::snprintf(buf, size, "\tld.global%s\t", 
                           ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(buf,size,", ");
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"+");
        res += bp; buf += bp; size -= bp;
        bp = src[1]->snprint(buf,size,"];\n");
        return res + bp;
    case PTX_LD_PARAM:
        bp = std::snprintf(buf, size, "\tld.param%s\t", ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(buf,size,", ");
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"];\n");
        return res + bp;
    case PTX_ST_GLOBAL:
        bp = std::snprintf(buf, size, "\tst.global%s\t", 
                           ptxTypeStr(dest->type));
        res += bp; buf += bp; size -= bp;
        bp = src[0]->snprint("[",buf,size,"+");
        res += bp; buf += bp; size -= bp;
        bp = src[1]->snprint(buf,size,"]");
        res += bp; buf += bp; size -= bp;
        bp = dest->snprint(", ",buf,size,";\n");
        return res + bp;
    case PTX_CVT:
        return snprintConvertOp(buf, size);
    case PTX_SETP_GE:
        return snprintLogicOp(buf, size);
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
        bp = snprintf(buf, size, "%4s",guardMod?"@":"@!");
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



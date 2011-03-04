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

#ifndef __PTXOPCODE_H
#define __PTXOPCODE_H

#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum 
{
    PTX_BRA,
    PTX_ADD,
    PTX_SUB,
    PTX_MUL,
    PTX_MAD,
    PTX_MAD_WIDE,
    PTX_DIV,
    PTX_REM,
    PTX_ST_GLOBAL,
    PTX_LD_GLOBAL,
    PTX_LD_PARAM,
    PTX_MOV,
    PTX_SETP_GE,
    PTX_EXIT
} PTXopcode;

#define PTX_MAX_OPERANDS (4)


const int ptxOperands(PTXopcode opcode);
const int ptxSrcOperands(PTXopcode opcode);
PTXopcode ptxOpcodeMap(cphvb_opcode opcode);

#ifdef __cplusplus
}
#endif

#endif

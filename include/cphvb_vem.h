/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
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

#ifndef __CPHVB_VEM_H
#define __CPHVB_VEM_H

#include <cphvb_type.h>
#include <cphvb_instruction.h>
#include <cphvb_opcode.h>


typedef struct
{
    cphvb_bool opcode[CPHVB_NO_OPCODES];//list of opcode support
    cphvb_bool type[CPHVB_NO_OPCODES];  //list of type support
} cphvb_support;


/* Codes for known components */
enum /* cphvb_comp */
{
    CPHVB_BRIDGE,
    CPHVB_VE_SIMPLE,
    CPHVB_VE_CUDA
};
typedef cphvb_intp cphvb_comp;

#endif

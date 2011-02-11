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

#ifndef __CPHVB_OPCODE_H
#define __CPHVB_OPCODE_H

#include "cphvb_type.h"

/* Codes for known oparations */
enum /* cphvb_opcode */
{
    CPHVB_ADD,
    CPHVB_SUB,
    CPHVB_MULT,
    CPHVB_DIV,
    CPHVB_MOD,
    CPHVB_NEG,
    CPHVB_MALLOC,
    CPHVB_FREE,
    CPHVB_READ,
    CPHVB_WRITE,
};

typedef cphvb_int32 cphvb_opcode;

#define CPHVB_MAX_NO_OPERANDS 3

#endif

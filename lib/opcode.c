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

#include "opcode.h"
#include "cphvb.h"

/* Number of operands for a given operation */
const int _operands[] = 
{ 
    [CPHVB_ADD] = 3,
    [CPHVB_SUB] = 3,
    [CPHVB_MULT] = 3,
    [CPHVB_DIV] = 3,
    [CPHVB_MOD] = 3,
    [CPHVB_NEG] = 2,
    [CPHVB_MALLOC] = 1,
    [CPHVB_FREE] = 1,
    [CPHVB_READ] = 2,
    [CPHVB_WRITE] = 2
};


/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int cphvb_operands(cphvb_opcode opcode)
{
    return _operands[opcode];
}

const char* const _opcode_text[] =
{
    [CPHVB_ADD] = "CPHVB_ADD",
    [CPHVB_SUB] = "CPHVB_SUB",
    [CPHVB_MULT] = "CPHVB_MULT",
    [CPHVB_DIV] = "CPHVB_DIV",
    [CPHVB_MOD] = "CPHVB_MOD",
    [CPHVB_NEG] = "CPHVB_NEG",
    [CPHVB_MALLOC] = "CPHVB_MALLOC",
    [CPHVB_FREE] = "CPHVB_FREE",
    [CPHVB_READ] = "CPHVB_READ",
    [CPHVB_WRITE] = "CPHVB_WRITE"
};


/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
const char* cphvb_opcode_text(cphvb_opcode opcode)
{
    return _opcode_text[opcode];
}

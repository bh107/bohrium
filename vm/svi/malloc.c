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

#include "svi.h"
#include "handler.h"
#include "util.h"
#include <stdlib.h>
#include <stdio.h>

cphvb_error svi_handle_malloc(cphvb_instruction* inst)
{
    if (inst->operand[0] > SVI_MAPSIZE-1)
        return CPHVB_OPERAND_UNKNOWN;
    size_t size = cphvb_nelements(inst->ndim, inst->shape) * 
        cphvb_typesize[inst->type[0]];
    void* ptr = malloc(size);
    if (ptr == NULL)
        return CPHVB_OUT_OF_MEMORY;
    svi_operand_map[inst->operand[0]] = ptr;
    return CPHVB_SUCCESS;
}

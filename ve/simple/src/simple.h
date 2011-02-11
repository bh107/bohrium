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

#ifndef __SVI_H
#define __SVI_H

#include <cphvb.h>
#include <cphvbutil.h>
#include <stddef.h>

#define SVI_MAPSIZE 128

// Global array that maps cphvb_operand's to memory pointers
void** svi_operand_map;



cphvb_error cphvb_ve_simple_init(void);

cphvb_error cphvb_ve_simple_execute(cphvb_int32 instruction_count,
                                    char* instruction_list);

cphvb_error cphvb_ve_simple_shutdown(void);

#endif

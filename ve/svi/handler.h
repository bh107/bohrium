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

#ifndef __SVI_HANDLER_H
#define __SVI_HANDLER_H
#include "svi.h"

cphvb_error svi_handle_add(cphvb_instruction*);
cphvb_error svi_handle_malloc(cphvb_instruction*);
cphvb_error svi_handle_free(cphvb_instruction*);
cphvb_error svi_handle_read(cphvb_instruction*);

cphvb_error svi_get_pointer(cphvb_operand, void**);

#endif

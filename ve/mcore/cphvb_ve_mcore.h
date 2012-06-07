/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __CPHVB_VE_MCORE_H
#define __CPHVB_VE_MCORE_H

#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT cphvb_error cphvb_ve_mcore_init(cphvb_component *self);

DLLEXPORT cphvb_error cphvb_ve_mcore_execute(cphvb_intp instruction_count,
                                   cphvb_instruction* instruction_list);

DLLEXPORT cphvb_error cphvb_ve_mcore_shutdown(void);

DLLEXPORT cphvb_error cphvb_ve_mcore_reg_func(char *fun, cphvb_intp *id);

#ifdef __cplusplus
}
#endif

#endif

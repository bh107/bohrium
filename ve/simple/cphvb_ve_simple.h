/*
 * Copyright 2012 Kenneth Skovhede <kenneth@hexad.dk>
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

#ifndef __CPHVB_SIMPLE_H
#define __CPHVB_SIMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT cphvb_error cphvb_ve_simple_init(cphvb_com *new_self);
DLLEXPORT cphvb_error cphvb_ve_simple_shutdown(void);
DLLEXPORT cphvb_error cphvb_ve_simple_reg_func(char *lib, char *fun, cphvb_intp *id);
DLLEXPORT cphvb_error cphvb_ve_simple_execute(cphvb_intp instruction_count,
                                    cphvb_instruction instruction_list[]);
DLLEXPORT cphvb_error cphvb_reduce(cphvb_userfunc* arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif

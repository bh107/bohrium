/*
 * Copyright 2012 Simon Andreas Frimann Lund <safl@safl.dk> 
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

#ifndef __CPHVB_COMPUTE_H
#define __CPHVB_COMPUTE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef cphvb_error (*computeloop)( cphvb_instruction*, cphvb_index, cphvb_index );

computeloop cphvb_compute_get( cphvb_instruction *instr );
cphvb_error cphvb_compute_apply( cphvb_instruction *instr );
cphvb_error cphvb_compute_reduce(cphvb_userfunc *arg, void* ve_arg);
cphvb_error cphvb_compute_random(cphvb_userfunc *arg, void* ve_arg);
cphvb_error cphvb_compute_matmul(cphvb_userfunc *arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif

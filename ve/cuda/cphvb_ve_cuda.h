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

#ifndef __CPHVB_VM_CUDA_H
#define __CPHVB_VM_CUDA_H

#include <cphvb.h>

#ifdef __cplusplus
/* C++ includes go here */
extern "C" {
#else
/* plain C includes go here */
#endif 

cphvb_error cphvb_ve_cuda_init(cphvb_intp *opcode_count,
                               cphvb_opcode opcode_list[],
                               cphvb_intp *datatype_count,
                               cphvb_type datatype_list[]);

cphvb_error cphvb_ve_cuda_execute(cphvb_intp instruction_count,
                                  cphvb_instruction instruction_list[]);

cphvb_error cphvb_ve_cuda_shutdown(void);


#ifdef __cplusplus 
}
#endif

#endif

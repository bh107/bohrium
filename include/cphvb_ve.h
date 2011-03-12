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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __CPHVB_VE_H
#define __CPHVB_VE_H

#include <cphvb.h>


/* Initialize the VE
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_ve_init)(cphvb_intp *opcode_count,
                                     cphvb_opcode opcode_list[CPHVB_MAX_NO_OPERANDS],
                                     cphvb_intp *datatype_count,
                                     cphvb_type datatype_list[CPHVB_NO_TYPES]);

/* Shutdown the VE, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_ve_shutdown)(void);

/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VE supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_ve_execute)(cphvb_intp count,
                                        cphvb_instruction inst_list[]);


#endif

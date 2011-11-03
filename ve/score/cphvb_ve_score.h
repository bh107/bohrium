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

#ifndef __CPHVB_VE_SCORE_H
#define __CPHVB_VE_SCORE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <cphvb.h>

cphvb_error cphvb_ve_score_init(cphvb_intp *opcode_count,
                                cphvb_opcode opcode_list[CPHVB_MAX_NO_OPERANDS],
                                cphvb_intp *datatype_count,
                                cphvb_type datatype_list[CPHVB_NO_TYPES]);

cphvb_error cphvb_ve_score_execute( cphvb_intp instruction_count,
                                    cphvb_instruction instruction_list[CPHVB_MAX_NO_OPERANDS]);

cphvb_error cphvb_ve_score_shutdown(void);

#ifdef __cplusplus
}
#endif 

#endif

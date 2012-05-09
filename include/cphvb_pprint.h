/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
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

#ifndef __CPHVB_PPRINT_H
#define __CPHVB_PPRINT_H

#include "cphvb_opcode.h"
#include "cphvb_array.h"
#include "cphvb_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pretty print an array.
 *
 * @param instr The array in question
 */
void cphvb_pprint_array( cphvb_array *array );

/* Pretty print an instruction.
 *
 * @param instr The instruction in question.
 */
void cphvb_pprint_instr( cphvb_instruction *instr );

void cphvb_pprint_instr_list( cphvb_instruction* instruction_list, cphvb_intp instruction_count, const char* txt );

/**
 * Pretty print a list of instructions
 * @param instruction_count Number of instruction in list.
 * @param instruction_list Array of instructions.
 */
void cphvb_pprint_bundle( cphvb_intp instruction_count, cphvb_instruction* instruction_list );

#ifdef __cplusplus
}
#endif

#endif

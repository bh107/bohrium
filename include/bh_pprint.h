/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __CPHVB_PPRINT_H
#define __CPHVB_PPRINT_H

#include "bh_opcode.h"
#include "bh_array.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Print a bh_array.
 *
 * @param array The array to print.
 */
DLLEXPORT void bh_pprint_array( bh_array *array );

/** Print a single bh_instruction.
 *
 * @param instr The instruction to print.
 */
DLLEXPORT void bh_pprint_instr( bh_instruction *instr );

/** Print a list of bh_instructions, prefixes with text.
 * 
 * @param instruction_list List of instructions to print.
 * @param instruction_count Number of bh_instruction in list.
 * @param txt String to prefix the printed output.
 */
DLLEXPORT void bh_pprint_instr_list( bh_instruction* instruction_list, bh_intp instruction_count, const char* txt );

/** Print a bundle of bh_instructions.
 * 
 * @param instruction_count Number of instruction in list.
 * @param instruction_list Array of instructions.
 */
DLLEXPORT void bh_pprint_bundle( bh_instruction* instruction_list, bh_intp instruction_count );

/** Print a coordinate.
 *
 * @param coord The coordinate to print
 * @param dims The number of dimensions in the coordinate.
 *
 */
DLLEXPORT void bh_pprint_coord( bh_index* coord, bh_index dims );

#ifdef __cplusplus
}
#endif

#endif

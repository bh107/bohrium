/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __CPHVB_PPRINT_H
#define __CPHVB_PPRINT_H

#include "cphvb_opcode.h"
#include "cphvb_array.h"
#include "cphvb_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Print a cphvb_array.
 *
 * @param array The array to print.
 */
DLLEXPORT void cphvb_pprint_array( cphvb_array *array );

/** Print a single cphvb_instruction.
 *
 * @param instr The instruction to print.
 */
DLLEXPORT void cphvb_pprint_instr( cphvb_instruction *instr );

/** Print a list of cphvb_instructions, prefixes with text.
 * 
 * @param instruction_list List of instructions to print.
 * @param instruction_count Number of cphvb_instruction in list.
 * @param txt String to prefix the printed output.
 */
DLLEXPORT void cphvb_pprint_instr_list( cphvb_instruction* instruction_list, cphvb_intp instruction_count, const char* txt );

/** Print a bundle of cphvb_instructions.
 * 
 * @param instruction_count Number of instruction in list.
 * @param instruction_list Array of instructions.
 */
DLLEXPORT void cphvb_pprint_bundle( cphvb_instruction* instruction_list, cphvb_intp instruction_count );

/** Print a coordinate.
 *
 * @param coord The coordinate to print
 * @param dims The number of dimensions in the coordinate.
 *
 */
DLLEXPORT void cphvb_pprint_coord( cphvb_index* coord, cphvb_index dims );

#ifdef __cplusplus
}
#endif

#endif

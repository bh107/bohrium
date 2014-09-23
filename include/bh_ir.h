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

#ifndef __BH_IR_H
#define __BH_IR_H

#include "bh_type.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
typedef struct
{
    //The list of Bohrium instructions
    bh_instruction *instr_list;
    //Number of instruction in the instruction list
    bh_intp ninstr;
    //Whether the BhIR did the memory allocation itself or not
    bool self_allocated;
} bh_ir;

/* Returns the total size of the BhIR including overhead (in bytes).
 *
 * @bhir    The BhIR in question
 * @return  Total size in bytes
 */
DLLEXPORT bh_intp bh_ir_totalsize(const bh_ir *bhir);

/* Creates a Bohrium Internal Representation (BhIR)
 * based on a instruction list. It will consist of one DAG.
 *
 * @bhir        The BhIR handle
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_ir_create(bh_ir *bhir, bh_intp ninstr,
                                const bh_instruction instr_list[]);

/* Destory a Bohrium Internal Representation (BhIR).
 *
 * @bhir        The BhIR handle
 */
DLLEXPORT void bh_ir_destroy(bh_ir *bhir);

/* Serialize a Bohrium Internal Representation (BhIR).
 *
 * @dest    The destination of the serialized BhIR
 * @bhir    The BhIR to serialize
 * @return  Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_ir_serialize(void *dest, const bh_ir *bhir);

/* De-serialize the BhIR (inplace)
 *
 * @bhir The BhIR in question
 */
DLLEXPORT void bh_ir_deserialize(bh_ir *bhir);

#ifdef __cplusplus
}
#endif

#endif


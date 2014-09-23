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

#include <bh.h>
#include <assert.h>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include "bh_ir.h"
#include "bh_vector.h"

/* Returns the total size of the BhIR including overhead (in bytes).
 *
 * @bhir    The BhIR in question
 * @return  Total size in bytes
 */
bh_intp bh_ir_totalsize(const bh_ir *bhir)
{
    bh_intp size = sizeof(bh_ir) + sizeof(bh_instruction)*bhir->ninstr;
    return size;
}

/* Creates a Bohrium Internal Representation (BhIR)
 * based on a instruction list. It will consist of one DAG.
 *
 * @bhir        The BhIR handle
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_ir_create(bh_ir *bhir, bh_intp ninstr,
                      const bh_instruction instr_list[])
{
    bh_intp instr_nbytes = sizeof(bh_instruction)*ninstr;

    //Make a copy of the instruction list
    bhir->instr_list = (bh_instruction*) bh_vector_create(instr_nbytes, ninstr, ninstr);
    if(bhir->instr_list == NULL)
        return BH_OUT_OF_MEMORY;
    memcpy(bhir->instr_list, instr_list, instr_nbytes);
    bhir->ninstr = ninstr;
    bhir->self_allocated = true;
    return BH_SUCCESS;
}

/* Destory a Bohrium Internal Representation (BhIR).
 *
 * @bhir        The BhIR handle
 */
void bh_ir_destroy(bh_ir *bhir)
{
    if(bhir->self_allocated)
    {
        bh_vector_destroy(bhir->instr_list);
    }
}


/* Serialize a Bohrium Internal Representation (BhIR).
 *
 * @dest    The destination of the serialized BhIR
 * @bhir    The BhIR to serialize
 * @return  Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_ir_serialize(void *dest, const bh_ir *bhir)
{
    //Serialize the static data of bh_ir
    bh_ir *b = (bh_ir *) dest;
    b->ninstr = bhir->ninstr;
    b->self_allocated = false;

    //Serialize the instr_list
    b->instr_list = (bh_instruction *) (b+1);
    memcpy(b->instr_list, bhir->instr_list,
           bhir->ninstr*sizeof(bh_instruction));

    //Convert to relative pointer address
    b->instr_list = (bh_instruction*)(((bh_intp)b->instr_list)-((bh_intp)(dest)));
    assert(b->instr_list >= 0);

    return BH_SUCCESS;
}


/* De-serialize the BhIR (inplace)
 *
 * @bhir The BhIR in question
 */
void bh_ir_deserialize(bh_ir *bhir)
{
    //Convert to absolut pointer address
    bhir->instr_list = (bh_instruction*)(((bh_intp)bhir->instr_list)+((bh_intp)(bhir)));
}

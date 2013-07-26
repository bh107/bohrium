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
#include "bh_adjmat.h"
#include "bh_boolmat.h"
#include <map>
#include <vector>


/* Creates an adjacency matrix based on a instruction list
 * where an index in the instruction list refer to a row or
 * a column index in the adjacency matrix.
 *
 * @adjmat      The adjacency matrix handle
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_adjmat_create_from_instr(bh_adjmat *adjmat, bh_intp ninstr,
                                     const bh_instruction instr_list[])
{
    bh_error e = bh_boolmat_create(&adjmat->m, ninstr);
    if(e != BH_SUCCESS)
        return e;

    //Record over which instructions (identified by indexes in the instruction list)
    //are writting to a specific array. We use a std::vector since multiple instructions
    //may write to the same array.
    std::map<bh_base*, std::vector<bh_intp> > writes, reads;


    for(bh_intp i=0; i<ninstr; ++i)
    {
        const bh_instruction *inst = &instr_list[i];
        const bh_view *ops = bh_inst_operands((bh_instruction *)inst);
        int nops = bh_operands_in_instruction(inst);
        //Check for dependencies
        for(bh_intp j=1; j<nops; ++j)
        {
            if(bh_is_constant(&ops[j]))
                continue;//Ignore constants
            bh_base *base = bh_base_array(&ops[j]);
            //Find the instructions the i'th instruction depends on
            std::vector<bh_intp> &deps(writes[base]);
            if(deps.size() > 0)
            {
                //Fill the i'th row in the boolean matrix with the found dependencies
                bh_boolmat_fill_empty_row(&adjmat->m, i, deps.size(), &deps[0]);
            }
        }
        //
        bh_base *base = bh_base_array(&ops[0]);
        writes[base].push_back(i);
//        NOT FINISHED

    }
    return BH_SUCCESS;
}

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
#include <set>
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
    bh_error e = bh_boolmat_create(&adjmat->mT, ninstr);
    if(e != BH_SUCCESS)
        return e;

    //Record over which instructions (identified by indexes in the instruction list)
    //are reading to a specific array. We use a std::vector since multiple instructions
    //may read to the same array.
    std::map<bh_base*, std::vector<bh_intp> > reads;

    //Record over the last instruction (identified by indexes in the instruction list)
    //that wrote to a specific array.
    //We only need the most recent write instruction since that instruction will depend on
    //all preceding write instructions.
    std::map<bh_base*, bh_intp> writes;

//    bh_pprint_instr_list(instr_list, ninstr, "Batch");
    for(bh_intp i=0; i<ninstr; ++i)
    {
        const bh_instruction *inst = &instr_list[i];
        const bh_view *ops = bh_inst_operands((bh_instruction *)inst);
        int nops = bh_operands_in_instruction(inst);

        //Find the instructions that the i'th instruction depend on and insert them into
        //the sorted set 'deps'.
        std::set<bh_intp> deps;
        for(bh_intp j=0; j<nops; ++j)
        {
            if(bh_is_constant(&ops[j]))
                continue;//Ignore constants
            bh_base *base = bh_base_array(&ops[j]);
            //When we are accessing an array, we depend on the instruction that wrote
            //to it previously (if any).
            std::map<bh_base*, bh_intp>::iterator w = writes.find(base);
            if(w != writes.end())
                deps.insert(w->second);

        }
        //When we are writing to an array, we depend on all previous reads that hasn't
        //already been overwritten
        bh_base *base = bh_base_array(&ops[0]);
        std::vector<bh_intp> &r(reads[base]);
        deps.insert(r.begin(), r.end());

        //Now all previous reads is overwritten
        r.clear();

        //Fill the i'th row in the boolean matrix with the found dependencies
        if(deps.size() > 0)
        {
            std::vector<bh_intp> sorted_vector(deps.begin(), deps.end());
            e = bh_boolmat_fill_empty_row(&adjmat->mT, i, deps.size(), &sorted_vector[0]);
            if(e != BH_SUCCESS)
                return e;
        }

        //The i'th instruction is now the newest write to array 'ops[0]'
        writes[base] = i;
        //and among the reads to arrays 'ops[1:]'
        for(bh_intp j=1; j<nops; ++j)
        {
            if(bh_is_constant(&ops[j]))
                continue;//Ignore constants
            bh_base *base = bh_base_array(&ops[j]);
            reads[base].push_back(i);
        }
    }
    //Lets compute the transposed matrix
    return bh_boolmat_transpose(&adjmat->m, &adjmat->mT);
}


/* De-allocate the adjacency matrix
 *
 * @adjmat  The adjacency matrix in question
 */
void bh_adjmat_destroy(bh_adjmat *adjmat)
{
    bh_boolmat_destroy(&adjmat->m);
    bh_boolmat_destroy(&adjmat->mT);
}


/* Retrieves a reference to a row in the adjacency matrix, i.e retrieval of the
 * node indexes that depend on the row'th node.
 *
 * @adjmat    The adjacency matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @return    List of column indexes (output)
 */
const bh_intp *bh_adjmat_get_row(const bh_adjmat *adjmat, bh_intp row, bh_intp *ncol_idx)
{
    return bh_boolmat_get_row(&adjmat->m, row, ncol_idx);
}


/* Retrieves a reference to a column in the adjacency matrix, i.e retrieval of the
 * node indexes that the col'th node depend on.
 *
 * @adjmat    The adjacency matrix
 * @col       The index of the column
 * @nrow_idx  Number of row indexes (output)
 * @return    List of row indexes (output)
 */
const bh_intp *bh_adjmat_get_col(const bh_adjmat *adjmat, bh_intp col, bh_intp *nrow_idx)
{
    return bh_boolmat_get_row(&adjmat->mT, col, nrow_idx);
}

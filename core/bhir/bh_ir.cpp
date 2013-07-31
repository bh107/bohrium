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
#include "bh_ir.h"


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
    bhir->instr_list = (bh_instruction*) malloc(instr_nbytes);
    if(bhir->instr_list == NULL)
        return BH_OUT_OF_MEMORY;
    memcpy(bhir->instr_list, instr_list, instr_nbytes);
    bhir->ninstr = ninstr;

    //Allocate the DAG list
    bhir->dag_list = (bh_dag*) malloc(sizeof(bh_dag));
    if(bhir->dag_list == NULL)
        return BH_OUT_OF_MEMORY;
    bhir->ndag = 1;

    return BH_SUCCESS;
    //Initiate the first (and only) DAG
    bh_dag *dag = &bhir->dag_list[0];
    dag->node_map = (bh_intp*) malloc(sizeof(bh_intp));
    for(bh_intp i=0; i<ninstr; ++i)
        dag->node_map[0] = i;//A simple 1:1 map
    dag->nnode = ninstr;
    return bh_adjmat_create_from_instr(&dag->adjmat, ninstr, instr_list);
}


/* Destory a Bohrium Internal Representation (BhIR).
 *
 * @bhir        The BhIR handle
 */
void bh_ir_destroy(bh_ir *bhir)
{
    free(bhir->instr_list);
    free(bhir->dag_list);
}


/* Go to the root node of a specific DAG.
 *
 * @dag_node  The node that points to the DAG
 * @new_node  The BhIR node (output)
 * @return    BH_ERROR when the 'dag_node' dosn't points
 *            to a DAG in which case the 'new_node' is
 *            untouched, BH_SUCCESS otherwise.
*/
bh_error bh_ir_node_goto_dag(const bh_ir_node *dag_node,
                             bh_ir_node *new_node)
{
    bh_dag *dag = &dag_node->bhir->dag_list[dag_node->dag_idx];
    bh_intp idx = dag->node_map[dag_node->node_map_idx];
    if(idx > 0)//Positive indexes are instruction list references.
        return BH_ERROR;

    new_node->bhir = dag_node->bhir;
    new_node->node_map_idx = 0;//We go to the root of the DAG
    new_node->dag_idx = -1*idx-1;
    return BH_SUCCESS;
}


/* Iterate to the next node in a DAG (topological order)
 * NB: it will not iterate into a sub-DAG.
 *
 * @node     The BhIR node
 * @return   BH_ERROR when at the end, BH_SUCCESS otherwise
 */
bh_error bh_ir_node_next(bh_ir_node *node)
{
    if(node->dag_idx >= node->bhir->ndag)
        return BH_ERROR;

    bh_dag *dag = &node->bhir->dag_list[node->dag_idx];//Current DAG
    if(++node->node_map_idx >= dag->nnode)
    {
        node->node_map_idx = 0;
        if(++node->dag_idx >= node->bhir->ndag)
            return BH_ERROR;
    }
    return BH_SUCCESS;
}



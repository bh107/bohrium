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

#include "bh_adjmat.h"
#include "bh_type.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The Directed Acyclic Graph (DAG) dictates the dependencies
 * between the nodes, such that a topological order obeys the
 * precedence constraints of the nodes. That is, all dependencies
 * of a DAG node must be executed before itself. */
typedef struct
{
    //Number of nodes
    bh_intp nnode;

    //The Adjacency Matrix where each row or column index
    //represents a node in the DAG.
    bh_adjmat adjmat;

    //The Node Map that translate DAG nodes into a Bohrium instruction
    //or a sub-DAG. Given a row or column index from the Adjacency Matrix,
    //the Node Map maps to an index in the instruction list or an index
    //in the DAG list. A positive index refers to the instruction list
    //and a negative index refers to the DAG list (-1*index-1).
    bh_intp *node_map;

    //The tag that represents some additional information associated this DAG.
    bh_intp tag;
} bh_dag;

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
typedef struct
{
    //The list of Bohrium instructions
    bh_instruction *instr_list;
    //Number of instruction in the instruction list
    bh_intp ninstr;
    //The list of DAGs
    bh_dag *dag_list;
    //Number of DAGs in the DAG list
    bh_intp ndag;
} bh_ir;

/* The BhIR node, which might refer to a bh_instruction or a bh_dag. */
typedef struct
{
    //The BhIR to iterate
    const bh_ir *bhir;
    //The index in the dag list
    bh_intp      dag_idx;
    //The index in the node_map
    bh_intp      node_map_idx;
} bh_node;

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

/* Resets the node to the first node in the BhIR (topologically).
 *
 * @node    The node to reset
 * @bhir    The BhIR handle
*/
DLLEXPORT void bh_node_reset(bh_node *node, const bh_ir *bhir);

/* Iterate to the next node in a DAG (topological order)
 * NB: it will not iterate into a sub-DAG.
 *
 * @node     The BhIR node
 * @return   BH_ERROR when at the end, BH_SUCCESS otherwise
 */
DLLEXPORT bh_error bh_node_next(bh_node *node);

/* Splits the DAG into an updated version of itself and a new sub-DAG that
 * consist of the nodes in 'nodes_idx'. Instead of the nodes in sub-DAG,
 * the updated DAG will have a new node that represents the sub-DAG.
 *
 * @bhir        The BhIR node
 * @nnodes      Number of nodes in the new sub-DAG
 * @nodes_idx   The nodes in the original DAG that will constitute the new sub-DAG.
 *              NB: this list will be sorted inplace
 * @dag_idx     The original DAG to split and thus modified
 * @sub_dag_idx The new sub-DAG which will be overwritten. -1 indicates that the
 *              new sub-DAG should be appended the DAG list
 *
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
*/
DLLEXPORT bh_error bh_dag_split(bh_ir *bhir, bh_intp nnodes, bh_intp nodes_idx[],
                                bh_intp dag_idx, bh_intp sub_dag_idx);

/* Write the BhIR in the DOT format.
 *
 * @bhir      The graph to print
 * @filename  Name of the written dot file, the DAG number
 *            and ".dot" is appended the file name
 */
DLLEXPORT void bh_bhir2dot(const bh_ir* bhir, const char* filename);



#ifdef __cplusplus
}
#endif

#endif


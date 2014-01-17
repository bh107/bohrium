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

#ifndef __BH_ADJLIST_H
#define __BH_ADJLIST_H

#ifdef __cplusplus

#include "bh_instruction.h"
#include "bh_type.h"
#include "bh_error.h"
#include <set>
#include <vector>

/* The adjacency list (bh_adjlist) represents edges between
 * instructions <http://en.wikipedia.org/wiki/Adjacency_list>.
 * The adjacencies are directed such that each instruction index
 * has a list of adjacencies that represents dependencies.
 * Note that the adjacency list does not contain any Bohrium
 * instructions only indices to a possible instruction list.
 *
 * Compared to the adjacency matrix, the adjacency list only
 * supports one level of sub-DAGs.
 *
 * NB: the adjacency list is only accessible through C++
*/
typedef struct
{
    //The node's adjacencies (i.e. dependencies)
    std::set<bh_intp> adj;
    //The node's sub-DAG index
    bh_intp sub_dag;
} bh_adjlist_node;

typedef struct
{
    //The sub-DAG's adjacencies (i.e. dependencies)
    std::set<bh_intp> adj;
    //The nodes in the sub-DAG
    std::set<bh_intp> node;
} bh_adjlist_sub_dag;

typedef struct
{
    //The node list where each node refer to an instruction index
    std::vector<bh_adjlist_node> node;
    //The sub-DAG list where each node refer to an sub-DAG index
    std::vector<bh_adjlist_sub_dag> sub_dag;
} bh_adjlist;

/* Creates an adjacency list based on a instruction list
 * where an index in the instruction list refer to a row or
 * a column index in the adjacency matrix.
 *
 * @adjmat      The adjacency list in question
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * May throw exception (std::bad_alloc)
 */
DLLEXPORT void bh_adjlist_create_from_instr(bh_adjlist &adjlist, bh_intp ninstr,
                                            const bh_instruction instr_list[]);

/* Fills the dag_list in the ‘bhir’ based on the adjacency list ‘adjlist’
 * and the instruction list in the bhir.
 * NB: The dag_list within the bhir should be uninitialized (NULL).
 *
 * @adjmat  The adjacency list in question
 * @bhir    The bihr to update
 * May throw exception (std::bad_alloc)
 */
DLLEXPORT void bh_adjlist_fill_bhir(const bh_adjlist &adjlist, bh_ir *bhir);

#endif
#endif


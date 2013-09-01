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

#ifndef __BH_ADJMAT_H
#define __BH_ADJMAT_H

#include "bh_boolmat.h"
#include "bh_instruction.h"
#include "bh_type.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif


/* The adjacency matrix (bh_adjmat) represents edges between
 * nodes <http://en.wikipedia.org/wiki/Adjacency_matrix>.
 * The adjacencies are directed such that a row index represents
 * the source node and the column index represents the target node.
 * The index order always represents the topological order of the nodes
 * and thus a legal order of execution.
 *
 * In this implementation, we use sparse boolean matrices to store
 * the adjacencies but alternatives such as adjacency lists or
 * incidence lists is also a possibility.
*/
typedef struct
{
    //Adjacency matrix with a top-down direction, i.e. the adjacencies
    //of a row is its dependencies (who it depends on).
    bh_boolmat m;
    //Adjacency matrix with a bottom-up direction, i.e. the adjacencies
    //of a row is its dependees (who depends on it).
    bh_boolmat mT;//Note, it is simply a transposed copy of 'm'.
} bh_adjmat;


/* Creates an adjacency matrix based on a instruction list
 * where an index in the instruction list refer to a row or
 * a column index in the adjacency matrix.
 *
 * @adjmat      The adjacency matrix handle
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_adjmat_create_from_instr(bh_adjmat *adjmat, bh_intp ninstr,
                                               const bh_instruction instr_list[]);

/* De-allocate the adjacency matrix
 *
 * @adjmat  The adjacency matrix in question
 */
DLLEXPORT void bh_adjmat_destroy(bh_adjmat *adjmat);

/* Retrieves a reference to a row in the adjacency matrix, i.e retrieval of the
 * node indexes that depend on the row'th node.
 *
 * @adjmat    The adjacency matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @return    List of column indexes (output)
 */
DLLEXPORT const bh_intp *bh_adjmat_get_row(const bh_adjmat *adjmat, bh_intp row,
                                           bh_intp *ncol_idx);

/* Retrieves a reference to a column in the adjacency matrix, i.e retrieval of the
 * node indexes that the col'th node depend on.
 *
 * @adjmat    The adjacency matrix
 * @col       The index of the column
 * @nrow_idx  Number of row indexes (output)
 * @return    List of row indexes (output)
 */
DLLEXPORT const bh_intp *bh_adjmat_get_col(const bh_adjmat *adjmat, bh_intp col,
                                           bh_intp *nrow_idx);

#ifdef __cplusplus
}
#endif

#endif


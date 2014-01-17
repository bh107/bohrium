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
    //Number of rows (and columns) in the square adjacency matrix
    bh_intp nrows;
    //Adjacency matrix with a top-down direction, i.e. the adjacencies
    //of a row is its dependencies (who it depends on).
    bh_boolmat *m;
    //Adjacency matrix with a bottom-up direction, i.e. the adjacencies
    //of a row is its dependees (who depends on it).
    bh_boolmat *mT;//Note, it is simply a transposed copy of 'm'.
    //Whether the adjmat did the memory allocation itself or not
    bool self_allocated;
} bh_adjmat;

/* Returns the total size of the adjmat including overhead (in bytes).
 *
 * @adjmat  The adjmat matrix in question
 * @return  Total size in bytes
 */
DLLEXPORT bh_intp bh_adjmat_totalsize(const bh_adjmat *adjmat);

/* Creates an empty adjacency matrix (Square Matrix)
 * @nrows   Number of rows (and columns) in the matrix.
 * @return  The adjmat handle, or NULL when out-of-memory
 */
DLLEXPORT bh_adjmat *bh_adjmat_create(bh_intp nrows);

/* Finalize the adjacency matrix such that it is accessible through
 * bh_adjmat_fill_empty_row(), bh_adjmat_serialize(), etc.
 *
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_adjmat_finalize(bh_adjmat *adjmat);

/* De-allocate the adjacency matrix
 *
 * @adjmat  The adjacency matrix in question
 */
DLLEXPORT void bh_adjmat_destroy(bh_adjmat **adjmat);

/* Fills a empty row in the adjacency matrix where all
 * the preceding rows are empty as well. That is, registrate whom
 * the row'th node depends on in the DAG.
 * Hint: use this function to build a adjacency matrix from
 *       scratch by filling each row in an ascending order.
 * NB: The adjmat must have been finalized.
 *
 * @adjmat    The adjmat matrix
 * @row       The index to the empty row
 * @ncol_idx  Number of column indexes (i.e. number of dependencies)
 * @col_idx   List of column indexes (i.e. whom the node depends on)
 *            NB: this list will be sorted thus any order is acceptable
 * @return    Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_adjmat_fill_empty_row(bh_adjmat *adjmat,
                                            bh_intp row,
                                            bh_intp ncol_idx,
                                            const bh_intp col_idx[]);

/* Fills a empty column in the adjacency matrix where all
 * the preceding columns are empty as well. That is, registrate whom
 * in the DAG depends on the col'th node.
 * Hint: use this function to build a adjacency matrix from
 *       scratch by filling each column in an ascending order.
 * NB: The adjmat must have been finalized.
 *
 * @adjmat    The adjmat matrix
 * @col       The index to the empty column
 * @nrow_idx  Number of row indexes (i.e. number of dependencies)
 * @row_idx   List of row indexes (i.e. whom that depends on the node)
 *            NB: this list will be sorted thus any order is acceptable
 * @return    Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_adjmat_fill_empty_col(bh_adjmat *adjmat,
                                            bh_intp col,
                                            bh_intp nrow_idx,
                                            const bh_intp row_idx[]);

/* Makes a serialized copy of the adjmat
 * NB: The adjmat must have been finalized.
 *
 * @adjmat   The adjmat matrix in question
 * @dest     The destination of the serialized adjmat
 */
DLLEXPORT void bh_adjmat_serialize(void *dest, const bh_adjmat *adjmat);

/* De-serialize the adjmat (inplace)
 *
 * @adjmat  The adjmat in question
 */
DLLEXPORT void bh_adjmat_deserialize(bh_adjmat *adjmat);

/* Retrieves a reference to a row in the adjacency matrix, i.e retrieval of the
 * node indexes that depend on the row'th node.
 * NB: The adjmat must have been finalized.
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
 * NB: The adjmat must have been finalized.
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


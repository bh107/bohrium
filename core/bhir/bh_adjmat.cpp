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

/* Returns the total size of the adjmat including overhead (in bytes).
 *
 * @adjmat  The adjmat matrix in question
 * @return  Total size in bytes
 */
bh_intp bh_adjmat_totalsize(const bh_adjmat *adjmat)
{
    //The adjmat must be initiated fully
    assert(adjmat->m != NULL);
    assert(adjmat->mT != NULL);
    return sizeof(bh_adjmat) + bh_boolmat_totalsize(adjmat->m)
                             + bh_boolmat_totalsize(adjmat->mT);
}

/* Creates an empty adjacency matrix (Square Matrix). The matrix is
 * write-once using either the bh_adjmat_fill_empty_row() or the
 * bh_adjmat_fill_empty_col() exclusively. Before accessing
 * call bh_adjmat_finalize().
 *
 * @nrows   Number of rows (and columns) in the matrix.
 * @return  The adjmat handle, or NULL when out-of-memory
 */
bh_adjmat *bh_adjmat_create(bh_intp nrows)
{
    bh_adjmat *adjmat = (bh_adjmat *) malloc(sizeof(bh_adjmat));
    if(adjmat == NULL)
        return NULL;
    adjmat->nrows = nrows;
    adjmat->self_allocated = true;
    adjmat->m = NULL;
    adjmat->mT= NULL;
    return adjmat;
}

/* Finalize the adjacency matrix such that it is accessible through
 * bh_adjmat_fill_empty_row(), bh_adjmat_serialize(), etc.
 *
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_adjmat_finalize(bh_adjmat *adjmat)
{
    if(adjmat->m == NULL && adjmat->mT == NULL)
    {
        adjmat->m = bh_boolmat_create(adjmat->nrows);
        if(adjmat->m == NULL)
            return BH_OUT_OF_MEMORY;
        adjmat->mT = bh_boolmat_create(adjmat->nrows);
        if(adjmat->mT == NULL)
            return BH_OUT_OF_MEMORY;
    }
    else if(adjmat->m == NULL)
    {
        adjmat->m = bh_boolmat_transpose(adjmat->mT);
        if(adjmat->m == NULL)
            return BH_OUT_OF_MEMORY;
    }
    else if(adjmat->mT == NULL)
    {
        adjmat->mT = bh_boolmat_transpose(adjmat->m);
        if(adjmat->mT == NULL)
            return BH_OUT_OF_MEMORY;
    }
    return BH_SUCCESS;
}

/* De-allocate the adjacency matrix
 *
 * @adjmat  The adjacency matrix in question
 */
void bh_adjmat_destroy(bh_adjmat **adjmat)
{
    bh_adjmat *a = *adjmat;
    if(a->m != NULL)
        bh_boolmat_destroy(&a->m);
    if(a->mT != NULL)
        bh_boolmat_destroy(&a->mT);
    if(a->self_allocated)
        free(a);
    a = NULL;
}

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
bh_error bh_adjmat_fill_empty_row(bh_adjmat *adjmat,
                                  bh_intp row,
                                  bh_intp ncol_idx,
                                  const bh_intp col_idx[])
{
    if(adjmat->nrows <= row)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_row() - the row index "
        "is greater than the number of rows in the adjacency matrix\n");
        assert(adjmat->nrows > row);
        return BH_ERROR;
    }
    if(adjmat->nrows < ncol_idx)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_row() - ncol_idx is greater "
                        "than the number of rows in the adjacency matrix\n");
        assert(adjmat->nrows >= ncol_idx);
        return BH_ERROR;
    }
    //If the transposed matrix has been initiated, we know that the
    //matrix has been accessed already.
    if(adjmat->mT != NULL)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_row() - the adjacency "
                        "matrix has been finalized already\n");
        assert(adjmat->mT == NULL);
        return BH_ERROR;
    }
    if(adjmat->m == NULL)
    {
        adjmat->m = bh_boolmat_create(adjmat->nrows);
        if(adjmat->m == NULL)
            return BH_OUT_OF_MEMORY;
    }
    return bh_boolmat_fill_empty_row(adjmat->m, row, ncol_idx, col_idx);
}

/* Fills a empty column in the adjacency matrix where all
 * the preceding columns are empty as well. That is, registrate
 * whom in the DAG depends on the col'th node.
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
bh_error bh_adjmat_fill_empty_col(bh_adjmat *adjmat,
                                  bh_intp col,
                                  bh_intp nrow_idx,
                                  const bh_intp row_idx[])
{
    if(adjmat->nrows <= col)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_col() - the column index "
        "is greater than the number of columns in the adjacency matrix\n");
        assert(adjmat->nrows > col);
        return BH_ERROR;
    }
    if(adjmat->nrows < nrow_idx)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_col() - ncol_idx is greater "
                        "than the number of rows in the adjacency matrix\n");
        assert(adjmat->nrows >= nrow_idx);
        return BH_ERROR;
    }
    //If the transposed matrix has been initiated, we know that the
    //matrix has been finalized.
    if(adjmat->m != NULL)
    {
        fprintf(stderr, "bh_adjmat_fill_empty_col() - the adjacency "
                        "matrix has been finalized already\n");
        assert(adjmat->m == NULL);
        return BH_ERROR;
    }
    if(adjmat->mT == NULL)
    {
        adjmat->mT = bh_boolmat_create(adjmat->nrows);
        if(adjmat->mT == NULL)
            return BH_OUT_OF_MEMORY;
    }
    return bh_boolmat_fill_empty_row(adjmat->mT, col, nrow_idx, row_idx);
}

/* Makes a serialized copy of the adjmat.
 * NB: The adjmat must have been finalized.
 *
 * @adjmat   The adjmat matrix in question
 * @dest     The destination of the serialized adjmat
 */
void bh_adjmat_serialize(void *dest, const bh_adjmat *adjmat)
{
    assert(adjmat->m != NULL && adjmat->mT != NULL);

    bh_adjmat *head = (bh_adjmat*) dest;
    head->self_allocated = false;
    char *mem = (char*)(head+1);
    bh_boolmat_serialize(mem, adjmat->m);
    head->m = (bh_boolmat*) mem;
    mem += bh_boolmat_totalsize(adjmat->m);
    bh_boolmat_serialize(mem, adjmat->mT);
    head->mT = (bh_boolmat*) mem;

    //Convert to relative pointer address
    head->m  = (bh_boolmat*)(((bh_intp)head->m)-((bh_intp)(dest)));
    head->mT = (bh_boolmat*)(((bh_intp)head->mT)-((bh_intp)(dest)));
}


/* De-serialize the adjmat (inplace)
 *
 * @adjmat  The adjmat in question
 */
void bh_adjmat_deserialize(bh_adjmat *adjmat)
{
    //Convert to absolut pointer address
    adjmat->m  = (bh_boolmat*)(((bh_intp)adjmat->m)+((bh_intp)(adjmat)));
    adjmat->mT = (bh_boolmat*)(((bh_intp)adjmat->mT)+((bh_intp)(adjmat)));

    bh_boolmat_deserialize(adjmat->m);
    bh_boolmat_deserialize(adjmat->mT);
}


/* Retrieves a reference to a row in the adjacency matrix, i.e retrieval of the
 * node indexes that depend on the row'th node.
 * NB: The adjmat must have been finalized.
 *
 *
 * @adjmat    The adjacency matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @return    List of column indexes (output)
 */
const bh_intp *bh_adjmat_get_row(const bh_adjmat *adjmat, bh_intp row,
                                 bh_intp *ncol_idx)
{
    assert(adjmat->m != NULL && adjmat->mT != NULL);
    return bh_boolmat_get_row(adjmat->m, row, ncol_idx);
}


/* Retrieves a reference to a column in the adjacency matrix, i.e retrieval of the
 * node indexes that the col'th node depend on.
 * NB: The adjmat must have been finalized.
 *
 * @adjmat    The adjacency matrix
 * @col       The index of the column
 * @nrow_idx  Number of row indexes (output)
 * @return    List of row indexes (output)
 */
const bh_intp *bh_adjmat_get_col(const bh_adjmat *adjmat, bh_intp col, bh_intp *nrow_idx)
{
    assert(adjmat->m != NULL && adjmat->mT != NULL);
    return bh_boolmat_get_row(adjmat->mT, col, nrow_idx);
}

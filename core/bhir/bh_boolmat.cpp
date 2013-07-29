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
#include "bh_boolmat.h"

/* The Boolean matrix (bh_boolmat) is a squared matrix that
 * uses the sparse matrix representation Compressed sparse row (CSR).
 * Typically, it is used as an adjacency matrix when representing
 * dependencies in a Bohrium instruction batch.
 * Note, since we only handles Booleans we do not need the value list
 * typically used in CSR.
*/


/* Creates a empty squared boolean matrix.
 *
 * @boolmat  The boolean matrix handle
 * @nrow     Number of rows (and columns) in the matrix
 * @return   Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_boolmat_create(bh_boolmat *boolmat, bh_intp nrows)
{
    boolmat->nrows   = nrows;
    //We know the final size of row_ptr at creation time
    boolmat->row_ptr = (bh_intp*) malloc(sizeof(bh_intp)*nrows+1);
    memset(boolmat->row_ptr, 0, sizeof(bh_intp)*nrows);
    //The size of column index list is equal to the number of True values
    //in the boolean matrix, which is unknown at creation time
    boolmat->col_idx = bh_dynamic_list_create(sizeof(bh_intp), nrows*2);
    if(boolmat->row_ptr == NULL || boolmat->col_idx == NULL)
        return BH_OUT_OF_MEMORY;
    return BH_SUCCESS;
}


/* Fills a empty row in the boolean matrix where all
 * the following rows are empty as well.
 * Hint: use this function to build a Boolean matrix from
 * scratch by filling each row in an ascending order
 *
 * @boolmat   The boolean matrix
 * @row       The index to the empty row
 * @ncol_idx  Number of column indexes
 * @col_idx   List of column indexes (see CSR documentation)
 * @return    Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
bh_error bh_boolmat_fill_empty_row(bh_boolmat *boolmat, bh_intp row, bh_intp ncol_idx,
                                   const bh_intp col_idx[])
{
    if(!(0 <= row && row < boolmat->nrows))
    {
        fprintf(stderr, "ERR: bh_boolmat_fill_empty_row() - argument 'row' is out of range\n");
        return BH_ERROR;
    }

    bh_intp r = boolmat->row_ptr[row];
    for(bh_intp i=0; i<ncol_idx; ++i)
    {
        bh_intp c = bh_dynamic_list_append(boolmat->col_idx);
        if(c == -1)
            return BH_OUT_OF_MEMORY;
        //Since the boolean matrix should be filled in ascending row order
        //we know that the dynamic list grows synchronously
        assert(c == r+i);
        ((bh_intp*)boolmat->col_idx)[r+i] = col_idx[i];
    }
    boolmat->row_ptr[row+1] += ncol_idx;
    return BH_SUCCESS;
}



/* Retrieves a reference to a row in the boolean matrix
 *
 * @boolmat   The boolean matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @col_idx   List of column indexes (output)
 * @return    Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_boolmat_get_row(bh_boolmat *boolmat, bh_intp row, bh_intp *ncol_idx,
                            bh_intp *col_idx[])
{
    if(!(0 <= row && row < boolmat->nrows))
    {
        fprintf(stderr, "ERR: bh_boolmat_get_row() - argument 'row' is out of range\n");
        return BH_ERROR;
    }
    bh_intp s = boolmat->row_ptr[row];//Start index in col_idx
    bh_intp e = boolmat->row_ptr[row+1];//End index in col_idx
    *ncol_idx = e - s;
    *col_idx = &((bh_intp*)boolmat->col_idx)[s];
    return BH_SUCCESS;
}


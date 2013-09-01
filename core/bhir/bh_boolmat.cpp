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
#include <bh_vector.h>
#include <algorithm>

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
    assert(nrows > 0);
    boolmat->nrows = nrows;
    boolmat->non_zeroes = 0;
    //We know the final size of row_ptr at creation time
    boolmat->row_ptr = (bh_intp*) malloc(sizeof(bh_intp)*(nrows+1));
    if(boolmat->row_ptr == NULL)
        return BH_OUT_OF_MEMORY;
    memset(boolmat->row_ptr, 0, sizeof(bh_intp)*(nrows+1));
    //The size of column index list is equal to the number of True values
    //in the boolean matrix, which is unknown at creation time
    boolmat->col_idx = (bh_intp *) bh_vector_create(sizeof(bh_intp), 0,
                                                    nrows*2);
    if(boolmat->col_idx == NULL)
        return BH_OUT_OF_MEMORY;
    return BH_SUCCESS;
}

/* De-allocate the boolean matrix
 *
 * @boolmat  The boolean matrix in question
 */
void bh_boolmat_destroy(bh_boolmat *boolmat)
{
    if(boolmat->row_ptr != NULL)
        free(boolmat->row_ptr);
    if(boolmat->col_idx != NULL)
        bh_vector_destroy(boolmat->col_idx);
    boolmat->row_ptr = NULL;
    boolmat->col_idx = NULL;
    boolmat->nrows = 0;
    boolmat->non_zeroes = 0;
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
 *            NB: this list will be sorted thus any order is acceptable
 * @return    Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
bh_error bh_boolmat_fill_empty_row(bh_boolmat *boolmat,
                                   bh_intp row,
                                   bh_intp ncol_idx,
                                   const bh_intp col_idx[])
{
    if(ncol_idx <= 0)
        return BH_SUCCESS;

    if(!(0 <= row && row < boolmat->nrows))
    {
        fprintf(stderr, "ERR: bh_boolmat_fill_empty_row() - "
                "argument 'row' (%ld) is out of range\n", (long) row);
        return BH_ERROR;
    }
    if(boolmat->row_ptr[row] != boolmat->row_ptr[row+1])
    {
        fprintf(stderr, "ERR: bh_boolmat_get_row() - the following "
                "rows (rows greater than %ld) is not all zeroes\n", (long) row);
        return BH_ERROR;
    }

    //Since the boolean matrix should be filled in ascending row order
    //we know that the vector grows by appending the new column indexes
    bh_intp size = bh_vector_nelem(boolmat->col_idx);
    boolmat->col_idx = (bh_intp*) bh_vector_resize(boolmat->col_idx, size+ncol_idx);
    if(boolmat->col_idx == NULL)
        return BH_OUT_OF_MEMORY;

    //Lets copy the new column indexes to the end
    bh_intp *start = boolmat->col_idx+size;
    memcpy(start, col_idx, ncol_idx*sizeof(bh_intp));

    //Lets sort the column list in ascending order (inplace)
    std::sort(start, start+ncol_idx);

    //Lets update the preceding row pointers and the number of non-zeroes
    for(bh_intp i=row+1; i<=boolmat->nrows; ++i)
        boolmat->row_ptr[i] += ncol_idx;
    boolmat->non_zeroes += ncol_idx;
    assert(boolmat->non_zeroes == bh_vector_nelem(boolmat->col_idx));

/*
    printf("col idx:");
    for(bh_intp i=0; i<boolmat->non_zeroes; ++i)
        printf(" %ld", (long) boolmat->col_idx[i]);
    printf("\n");
    printf("row ptr:");
    for(bh_intp i=0; i<=boolmat->nrows; ++i)
        printf(" %ld", (long) boolmat->row_ptr[i]);
    printf("\n");
*/
    return BH_SUCCESS;
}


/* Retrieves a reference to a row in the boolean matrix
 *
 * @boolmat   The boolean matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @return    List of column indexes (output)
 */
const bh_intp *bh_boolmat_get_row(const bh_boolmat *boolmat,
                                  bh_intp row, bh_intp *ncol_idx)
{
    if(!(0 <= row && row < boolmat->nrows))
    {
        fprintf(stderr, "ERR: bh_boolmat_get_row() - "
                "argument 'row' (%ld) is out of range\n", (long) row);
        return NULL;
    }
    bh_intp s = boolmat->row_ptr[row];//Start index in col_idx
    bh_intp e = boolmat->row_ptr[row+1];//End index in col_idx
    *ncol_idx = e - s;
    return &boolmat->col_idx[s];
}


/* Returns a transposed copy
 *
 * @out     The output matrix
 * @in      The input matrix
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_boolmat_transpose(bh_boolmat *out, const bh_boolmat *in)
{
    bh_error e = bh_boolmat_create(out, in->nrows);
    if(e != BH_SUCCESS)
        return e;

    //Lets compute the row_ptr by creating a histogram
    for(bh_intp i=0; i<in->non_zeroes; ++i)
        ++out->row_ptr[in->col_idx[i]];
    //and then scanning it.
    bh_intp t[] = {0,0};
    t[0] = out->row_ptr[0];
    out->row_ptr[0] = 0;
    for(bh_intp i=1; i<= out->nrows; ++i)
    {
        t[i%2] = out->row_ptr[i];
        out->row_ptr[i] = t[(i-1)%2] + out->row_ptr[i-1];
    }

    //The size of col_idx equals the number of non-zero values
    out->col_idx = (bh_intp*) bh_vector_resize(out->col_idx, in->non_zeroes);
    out->non_zeroes = in->non_zeroes;

    //Lets compute the col_idx
    bh_intp counter[out->nrows];
    memset(counter, 0, sizeof(bh_intp)*out->nrows);
    for(bh_intp i=0; i < out->nrows; ++i)
    {
        for(bh_intp j=in->row_ptr[i]; j < in->row_ptr[i+1]; ++j)
        {
            bh_intp in_col_idx = in->col_idx[j];
            bh_intp pos = out->row_ptr[in_col_idx] + counter[in_col_idx];
            out->col_idx[pos] = i;
            ++counter[in_col_idx];
        }
    }
    return BH_SUCCESS;
}

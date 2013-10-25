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

/* Returns the total size of the boolmat including overhead (in bytes).
 *
 * @boolmat  The boolean matrix in question
 * @return   Total size in bytes
 */
bh_intp bh_boolmat_totalsize(const bh_boolmat *boolmat)
{
    return sizeof(bh_boolmat) + sizeof(bh_intp)*(boolmat->nrows+1) +
           bh_vector_totalsize(boolmat->col_idx);
}

/* Creates a empty squared boolean matrix.
 *
 * @nrow   Number of rows (and columns) in the matrix
 * @return The boolean matrix handle, or NULL when out-of-memory
 */
bh_boolmat *bh_boolmat_create(bh_intp nrows)
{
    bh_boolmat *ret = (bh_boolmat*) malloc(sizeof(bh_boolmat));
    if(ret == NULL)
        return ret;

    assert(nrows > 0);
    ret->nrows = nrows;
    ret->non_zeroes = 0;
    ret->self_allocated = true;
    //We know the final size of row_ptr at creation time
    ret->row_ptr = (bh_intp*) malloc(sizeof(bh_intp)*(nrows+1));
    if(ret->row_ptr == NULL)
        return NULL;
    memset(ret->row_ptr, 0, sizeof(bh_intp)*(nrows+1));
    //The size of column index list is equal to the number of True values
    //in the boolean matrix, which is unknown at creation time
    ret->col_idx = (bh_intp *) bh_vector_create(sizeof(bh_intp), 0,
                                                    nrows*2);
    if(ret->col_idx == NULL)
        return NULL;
    return ret;
}

/* De-allocate the boolean matrix
 *
 * @boolmat  The boolean matrix in question
 */
void bh_boolmat_destroy(bh_boolmat **boolmat)
{
    bh_boolmat *b = *boolmat;
    if(b->self_allocated)
    {
        if(b->row_ptr != NULL)
            free(b->row_ptr);
        if(b->col_idx != NULL)
            bh_vector_destroy(b->col_idx);
        b->row_ptr = NULL;
        b->col_idx = NULL;
        b->nrows = 0;
        b->non_zeroes = 0;
        free(b);
    }
    b = NULL;
}

/* Makes a serialized copy of the boolmat
 *
 * @boolmat  The boolean matrix in question
 * @dest     The destination of the serialized bolmat
 */
void bh_boolmat_serialize(void *dest, const bh_boolmat *boolmat)
{
    bh_boolmat *head = (bh_boolmat*) dest;
    head->nrows = boolmat->nrows;
    head->non_zeroes = boolmat->non_zeroes;
    head->self_allocated = false;

    char *body = ((char*)dest) + sizeof(bh_boolmat);
    memcpy(body, boolmat->row_ptr, sizeof(bh_intp)*(boolmat->nrows+1));
    head->row_ptr = (bh_intp*) body;

    body += sizeof(bh_intp)*(boolmat->nrows+1);
    memcpy(body, bh_vector_vector2memblock(boolmat->col_idx),
                 bh_vector_totalsize(boolmat->col_idx));
    head->col_idx = (bh_intp*) bh_vector_memblock2vector(body);

    //Convert to raletive pointer address
    head->row_ptr = (bh_intp*)(((bh_intp)head->row_ptr)-((bh_intp)(dest)));
    head->col_idx = (bh_intp*)(((bh_intp)head->col_idx)-((bh_intp)(dest)));
    assert(head->row_ptr >= 0);
    assert(head->col_idx >= 0);
}

/* De-serialize the boolmat (inplace)
 *
 * @boolmat  The boolean matrix in question
 */
void bh_boolmat_deserialize(bh_boolmat *boolmat)
{
    //Convert to absolut pointer address
    boolmat->row_ptr = (bh_intp*)(((bh_intp)boolmat->row_ptr)+((bh_intp)(boolmat)));
    boolmat->col_idx = (bh_intp*)(((bh_intp)boolmat->col_idx)+((bh_intp)(boolmat)));
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
 * @in      The input matrix
 * @return  The transposed boolmat or NULL on out-of-memory
 */
bh_boolmat *bh_boolmat_transpose(const bh_boolmat *in)
{
    bh_boolmat *ret = bh_boolmat_create(in->nrows);
    if(ret == NULL)
        return NULL;

    //Lets compute the row_ptr by creating a histogram
    for(bh_intp i=0; i<in->non_zeroes; ++i)
        ++ret->row_ptr[in->col_idx[i]];
    //and then scanning it.
    bh_intp t[] = {0,0};
    t[0] = ret->row_ptr[0];
    ret->row_ptr[0] = 0;
    for(bh_intp i=1; i<= ret->nrows; ++i)
    {
        t[i%2] = ret->row_ptr[i];
        ret->row_ptr[i] = t[(i-1)%2] + ret->row_ptr[i-1];
    }

    //The size of col_idx equals the number of non-zero values
    ret->col_idx = (bh_intp*) bh_vector_resize(ret->col_idx, in->non_zeroes);
    ret->non_zeroes = in->non_zeroes;

    //Lets compute the col_idx
    bh_intp counter[ret->nrows];
    memset(counter, 0, sizeof(bh_intp)*ret->nrows);
    for(bh_intp i=0; i < ret->nrows; ++i)
    {
        for(bh_intp j=in->row_ptr[i]; j < in->row_ptr[i+1]; ++j)
        {
            bh_intp in_col_idx = in->col_idx[j];
            bh_intp pos = ret->row_ptr[in_col_idx] + counter[in_col_idx];
            ret->col_idx[pos] = i;
            ++counter[in_col_idx];
        }
    }
    return ret;
}

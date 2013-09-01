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

#ifndef __BH_BOOLMAT_H
#define __BH_BOOLMAT_H

#include "bh_error.h"
#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The Boolean matrix (bh_boolmat) is a squared matrix that
 * uses the sparse matrix representation Compressed sparse row (CSR).
 * Typically, it is used as an adjacency matrix when representing
 * dependencies in a Bohrium instruction batch.
 * Note, since we only handles Booleans we do not need the value list
 * typically used in CSR. */
typedef struct
{
    //Number of rows (and columns) in the matrix
    bh_intp nrows;
    //List of row pointers but with a extra dummy pointer at the end of the list
    //that points to the last column index (see CSR documentation)
    bh_intp *row_ptr;
    //List of column indexes (see CSR documentation)
    bh_intp *col_idx;
    //Number of non-zeroes in the matrix
    bh_intp non_zeroes;
} bh_boolmat;

/* Creates a empty squared boolean matrix.
 *
 * @boolmat  The boolean matrix handle
 * @nrow     Number of rows (and columns) in the matrix
 * @return   Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_boolmat_create(bh_boolmat *boolmat, bh_intp nrows);

/* De-allocate the boolean matrix
 *
 * @boolmat  The boolean matrix in question
 */
DLLEXPORT void bh_boolmat_destroy(bh_boolmat *boolmat);

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
 * @return    Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_boolmat_fill_empty_row(bh_boolmat *boolmat,
                                             bh_intp row,
                                             bh_intp ncol_idx,
                                             const bh_intp col_idx[]);

/* Retrieves a reference to a row in the boolean matrix
 *
 * @boolmat   The boolean matrix
 * @row       The index to the row
 * @ncol_idx  Number of column indexes (output)
 * @return    List of column indexes (output)
 */
DLLEXPORT const bh_intp *bh_boolmat_get_row(const bh_boolmat *boolmat,
                                            bh_intp row,
                                            bh_intp *ncol_idx);


/* Returns a transposed copy
 *
 * @out  The output matrix
 * @in   The input matrix
 * @return    Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_boolmat_transpose(bh_boolmat *out, const bh_boolmat *in);

#ifdef __cplusplus
}
#endif

#endif


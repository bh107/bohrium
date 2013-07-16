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
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Reduce nDarray info to a base shape
 *
 * Remove dimentions that just indicate orientation in a
 * higher dimention (Py:newaxis)
 *
 * @ndim          Number of dimentions
 * @shape[]       Number of elements in each dimention.
 * @stride[]      Stride in each dimention.
 * @base_ndim     Placeholder for base number of dimentions
 * @base_shape    Placeholder for base number of elements in each dimention.
 * @base_stride   Placeholder for base stride in each dimention.
 */
void bh_base_shape(bh_intp ndim,
                      const bh_index shape[],
                      const bh_index stride[],
                      bh_intp* base_ndim,
                      bh_index* base_shape,
                      bh_index* base_stride)
{
    *base_ndim = 0;
    for (int i = 0; i < ndim; ++i)
    {
        // skipping (shape[i] == 1 && stride[i] == 0)
        if (shape[i] != 1 || stride[i] != 0)
        {
            base_shape[*base_ndim] = shape[i];
            base_stride[*base_ndim] = stride[i];
            ++(*base_ndim);
        }
    }
}

/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity.
 */
bool bh_is_continuous(bh_intp ndim,
                         const bh_index shape[],
                         const bh_index stride[])
{
    bh_intp my_ndim = 0;
    bh_index my_shape[BH_MAXDIM];
    bh_index my_stride[BH_MAXDIM];
    bh_base_shape(ndim, shape, stride, &my_ndim, my_shape, my_stride);
    for (int i = 0; i < my_ndim - 1; ++i)
    {
        if (my_shape[i+1] != my_stride[i])
            return true;
    }
    if (my_stride[my_ndim-1] != 1)
        return false;

    return true;
}

/* Number of non-broadcasted elements in a given view
 *
 * @view    The view in question.
 * @return  Number of elements.
 */
bh_index bh_nelements_nbcast(const bh_view *view)
{
    bh_index res = 1;
    for (int i = 0; i < view->ndim; ++i)
    {
        if(view->stride[i] > 0)
            res *= view->shape[i];
    }
    return res;
}

/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
bh_index bh_nelements(bh_intp ndim,
                            const bh_index shape[])
{
    bh_index res = 1;
    for (int i = 0; i < ndim; ++i)
    {
        res *= shape[i];
    }
    return res;
}

/* Size of the base array in bytes
 *
 * @base    The base in question
 * @return  The size of the base array in bytes
 */
bh_index bh_base_size(const bh_base *base)
{
    return base->nelem * bh_type_size(base->type);
}

/* Calculate the dimention boundries for shape
 *
 * @ndim      Number of dimentions
 * @shape[]   Number of elements in each dimention.
 * @dimbound  Placeholder for dimbound (return
 */
void bh_dimbound(bh_intp ndim,
                    const bh_index shape[],
                    bh_index* dimbound)
{
    dimbound[ndim -1] = shape[ndim -1];
    for (int i = ndim -2 ; i >= 0; --i)
    {
        dimbound[i] = dimbound[i+1] * shape[i];
    }
}

/* Set the view stride to contiguous row-major
 *
 * @view    The view in question
 * @return  The total number of elements in view
 */
bh_intp bh_set_contiguous_stride(bh_view *view)
{
    bh_intp s = 1;
    for(bh_intp i=view->ndim-1; i >= 0; --i)
    {
        view->stride[i] = s;
        s *= view->shape[i];
    }
    return s;
}

/* Updates the view with the complete base
 *
 * @view    The view to update (in-/out-put)
 * @base    The base assign to the view
 * @return  The total number of elements in view
 */
void bh_assign_complete_base(bh_view *view, bh_base *base)
{
    view->base = base;
    view->ndim = 1;
    view->start = 0;
    view->shape[0] = view->base->nelem;
    view->stride[0] = 1;
}

/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
bh_index bh_calc_offset(bh_intp ndim,
                              const bh_index shape[],
                              const bh_index stride[],
                              const bh_index element)
{
    bh_index offset = 0;
    bh_index dimIndex;
    bh_intp i;
    for (i = 0; i < ndim; ++i)
    {
        dimIndex = element % bh_nelements(ndim - i, &shape[i]);
        if (i != ndim - 1)
            dimIndex = dimIndex / bh_nelements(ndim - (i+1), &shape[i+1]);
        offset += dimIndex * stride[i];
    }
    return offset;
}

/* Set the data pointer for the view.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @view   The view in question
 * @data   The new data pointer
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_set(bh_view* view, bh_data_ptr data)
{
    bh_base* base;

    if(view == NULL)
    {
        fprintf(stderr, "Attempt to set data pointer for a null view\n");
        return BH_ERROR;
    }

    base = bh_base_array(view);

    if(base->data != NULL && data != NULL)
    {
        fprintf(stderr, "Attempt to set data pointer an array with existing data pointer\n");
        return BH_ERROR;
    }

    base->data = data;

    return BH_SUCCESS;
}

/* Get the data pointer for the view.
 *
 * @view    The view in question
 * @result  Output data pointer
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_get(bh_view* view, bh_data_ptr* result)
{
    bh_base* base;

    if(view == NULL)
    {
        fprintf(stderr, "Attempt to get data pointer for a null view\n");
        return BH_ERROR;
    }

    base = bh_base_array(view);

    *result = base->data;

    return BH_SUCCESS;
}

/* Allocate data memory for the given base if not already allocated.
 * For convenience, the base is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
bh_error bh_data_malloc(bh_base* base)
{
    bh_intp bytes;

    if(base == NULL)
        return BH_SUCCESS;

    if(base->data != NULL)
        return BH_SUCCESS;

    bytes = bh_base_size(base);
    if(bytes == 0)//We allow zero sized arrays.
        return BH_SUCCESS;

    if(bytes < 0)
        return BH_ERROR;

    base->data = bh_memory_malloc(bytes);
    if(base->data == NULL)
    {
        int errsv = errno;//mmap() sets the errno.
        printf("bh_data_malloc() could not allocate a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return BH_OUT_OF_MEMORY;
    }

    return BH_SUCCESS;
}

/* Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_free(bh_base* base)
{
    bh_intp bytes;

    if(base == NULL)
        return BH_SUCCESS;

    if(base->data == NULL)
        return BH_SUCCESS;

    bytes = bh_base_size(base);

    if(bh_memory_free(base->data, bytes) != 0)
    {
        int errsv = errno;//munmmap() sets the errno.
        printf("bh_data_free() could not free a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return BH_ERROR;
    }
    base->data = NULL;
    return BH_SUCCESS;
}


/* Retrive the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
bh_view *bh_inst_operands(bh_instruction *instruction)
{
    if (instruction->opcode == BH_USERFUNC)
        return (bh_view *) instruction->userfunc->operand;
    else
        return (bh_view *) &instruction->operand;
}

/* Determines whether the base array is a scalar.
 *
 * @view The view
 * @return The boolean answer
 */
bool bh_is_scalar(const bh_view* view)
{
    return bh_base_array(view)->nelem == 1;
}

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
bool bh_is_constant(const bh_view* o)
{
    return (o->base == NULL);
}

/* Flag operand as a constant
 *
 * @o      The operand
 */
void bh_flag_constant(bh_view* o)
{
    o->base = NULL;
}

/* Determines whether the two views are the same
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
bool bh_same_view(const bh_view* a, const bh_view* b)
{
    if (a == b)
        return true;
    if (bh_base_array(a) != bh_base_array(b))
        return false;
    if (memcmp(((char*)a)+sizeof(bh_view*),
               ((char*)b)+sizeof(bh_view*),
               sizeof(bh_view)-sizeof(bh_view*)-sizeof(bh_data_ptr)))
        return false;
    return true;
}

inline int gcd(int a, int b)
{
    int c = a % b;
    while(c != 0)
    {
        a = b;
        b = c;
        c = a % b;
    }
    return b;
}
/* Determines whether two views access some of the same data points
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
bool bh_disjoint_views(const bh_view *a, const bh_view *b)
{
    if (bh_is_constant(a) || bh_is_constant(b)) // One is a constant
        return true;
    if(bh_base_array(a) != bh_base_array(b)) //different base
        return true;
    if(a->ndim != b->ndim) // we dont handle views of differenr dimensions yet
        return false;

    int astart = a->start;
    int bstart = b->start;
    int stride = 1;
    for (int i = 0; i < a->ndim; ++i)
    {
        stride = gcd(a->stride[i], b->stride[i]);
        int as = astart / stride;
        int bs = bstart / stride;
        int ae = as + a->shape[i] * (a->stride[i]/stride);
        int be = bs + b->shape[i] * (b->stride[i]/stride);
        if (ae <= bs || be <= as)
            return true;
        astart %= stride;
        bstart %= stride;
    }
    if (stride > 1 && a->start % stride != b->start % stride)
        return true;
    return false;
}

/* Determines whether two views overlap.
 * NB: This functions may return True on non-overlapping views.
 *     But will always return False on overlapping views.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
bool bh_view_overlap(const bh_view *a, const bh_view *b)
{
    if(bh_is_constant(a) || bh_is_constant(b))
        return false;

    if(bh_base_array(a) != bh_base_array(b))
        return false;

    bh_intp a_nelem = bh_nelements(a->ndim, a->shape);
    bh_intp b_nelem = bh_nelements(b->ndim, b->shape);

    if(a_nelem <= 0 || b_nelem <= 0)
        return false;

    //Check for obvious data overlap
    bh_intp a_end = a->start + a_nelem;
    bh_intp b_end = b->start + b_nelem;
    if(a->start <= b->start && b->start < a_end)
        return true;
    if(a->start <= b_end && b_end < a_end)
        return true;
    if(b->start <= a->start && a->start < b_end)
        return true;
    if(b->start <= a_end && a_end < b_end)
        return true;

    return false;
}

/* Determines whether two views are identical and points
 * to the same base array.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
bool bh_view_identical(const bh_view *a, const bh_view *b)
{
    int i;
    if(bh_is_constant(a) || bh_is_constant(b))
        return false;
    if(a->base != b->base)
        return false;
    if(a->ndim != b->ndim)
        return false;
    for(i=0; i<a->ndim; ++i)
    {
        if(a->shape[i] != b->shape[i])
            return false;
        if(a->stride[i] != b->stride[i])
            return false;
    }
    return true;
}

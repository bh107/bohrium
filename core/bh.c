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

/* Size of the array data
 *
 * @array    The array in question
 * @return   The size of the array data in bytes
 */
bh_index bh_array_size(const bh_array *array)
{
    const bh_array *b = bh_base_array(array);
    return bh_nelements(b->ndim, b->shape) * 
           bh_type_size(b->type);
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

/* Set the array stride to contiguous row-major
 *
 * @array    The array in question
 * @return   The total number of elements in array
 */
bh_intp bh_set_contiguous_stride(bh_array *array)
{
    bh_intp s = 1;
    for(bh_intp i=array->ndim-1; i >= 0; --i)
    {    
        array->stride[i] = s;
        s *= array->shape[i];
    }
    return s;
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

/* Find the base array for a given array/view
 *
 * @view   Array/view in question
 * @return The Base array
 */
bh_array* bh_base_array(const bh_array* view)
{
    if(view->base == NULL)
    {
        return (bh_array*)view;
    }
    else
    {
        assert(view->base->base == NULL);
        return view->base;
    }
}

/* Set the data pointer for the array.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @array The array in question
 * @data The new data pointer
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_set(bh_array* array, bh_data_ptr data)
{
    bh_array* base;

    if(array == NULL)
    {
        fprintf(stderr, "Attempt to set data pointer for a null array\n");
        return BH_ERROR;
    }

    base = bh_base_array(array);

    if(base->data != NULL && data != NULL)
    {
        fprintf(stderr, "Attempt to set data pointer an array with existing data pointer\n");
        return BH_ERROR;
    }

    base->data = data;

    return BH_SUCCESS;
}

/* Get the data pointer for the array.
 *
 * @array The array in question
 * @result Output area
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_get(bh_array* array, bh_data_ptr* result)
{
    bh_array* base;

    if(array == NULL)
    {
        fprintf(stderr, "Attempt to get data pointer for a null array\n");
        return BH_ERROR;
    }

    base = bh_base_array(array);

    *result = base->data;

    return BH_SUCCESS;
}

/* Allocate data memory for the given array if not already allocated.
 * If @array is a view, the data memory for the base array is allocated.
 * NB: It does NOT initiate the memory.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_data_malloc(bh_array* array)
{
    bh_intp bytes;
    bh_array* base;

    if(array == NULL)
        return BH_SUCCESS;

    base = bh_base_array(array);

    if(base->data != NULL)
        return BH_SUCCESS;

    bytes = bh_array_size(base);
    if(bytes <= 0)
        return BH_SUCCESS;

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

/* Frees data memory for the given array.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_data_free(bh_array* array)
{
    bh_intp bytes;
    bh_array* base;

    if(array == NULL)
        return BH_SUCCESS;

    base = bh_base_array(array);

    if(base->data == NULL)
        return BH_SUCCESS;

    bytes = bh_array_size(base);

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
bh_array **bh_inst_operands(const bh_instruction *instruction)
{
    if (instruction->opcode == BH_USERFUNC)
        return (bh_array **) instruction->userfunc->operand;
    else
        return (bh_array **) instruction->operand;
}


/* Retrive the operand type of a instruction.
 *
 * @instruction  The instruction in question
 * @operand_no Number of the operand in question
 * @return The operand type
 */
bh_type bh_type_operand(const bh_instruction *instruction,
                              bh_intp operand_no)
{
    bh_array **operands = bh_inst_operands(instruction);
    bh_array *operand = operands[operand_no];

    if (bh_is_constant(operand))
        return instruction->constant.type;
    else
        return operand->type;
}

/* Determines whether two arrays overlap.
 * NB: This functions may return True on non-overlapping arrays. 
 *     But will always return False on overlapping arrays.
 * 
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool bh_array_overlap(const bh_array *a, const bh_array *b)
{
    if(a == NULL || b == NULL)
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

/* Determines whether the array is a scalar or a broadcast view of a scalar.
 *
 * @array The array
 * @return The boolean answer
 */
bool bh_is_scalar(const bh_array* array)
{
    return (bh_base_array(array)->ndim == 0) ||
        (bh_base_array(array)->ndim == 1 && bh_base_array(array)->shape[0] == 1);
}

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
bool bh_is_constant(const bh_array* o)
{
    return (o == NULL);
}

/* Determines whether the two views are the same
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool bh_same_view(const bh_array* a, const bh_array* b)
{
    if (a == b)
        return true;
    if (bh_base_array(a) != bh_base_array(b))
        return false;
    if (memcmp(((char*)a)+sizeof(bh_array*),
               ((char*)b)+sizeof(bh_array*),
               sizeof(bh_array)-sizeof(bh_array*)-sizeof(bh_data_ptr)))
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
/* Determines whether two array(views)s access some of the same data points
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool bh_disjoint_views(const bh_array *a, const bh_array *b)
{
    if (a == NULL || b == NULL) // One is a constant 
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

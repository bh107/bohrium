/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>
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
void cphvb_base_shape(cphvb_intp ndim,
                      const cphvb_index shape[],
                      const cphvb_index stride[],
                      cphvb_intp* base_ndim,
                      cphvb_index* base_shape,
                      cphvb_index* base_stride)
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
bool cphvb_is_continuous(cphvb_intp ndim,
                         const cphvb_index shape[],
                         const cphvb_index stride[])
{
    cphvb_intp my_ndim = 0;
    cphvb_index my_shape[CPHVB_MAXDIM];
    cphvb_index my_stride[CPHVB_MAXDIM];
    cphvb_base_shape(ndim, shape, stride, &my_ndim, my_shape, my_stride);
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
cphvb_index cphvb_nelements(cphvb_intp ndim,
                            const cphvb_index shape[])
{
    cphvb_index res = 1;
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
cphvb_index cphvb_array_size(const cphvb_array *array)
{
    const cphvb_array *b = cphvb_base_array(array);
    return cphvb_nelements(b->ndim, b->shape) * 
           cphvb_type_size(b->type);
}

/* Calculate the dimention boundries for shape
 *
 * @ndim      Number of dimentions
 * @shape[]   Number of elements in each dimention.
 * @dimbound  Placeholder for dimbound (return
 */
void cphvb_dimbound(cphvb_intp ndim,
                    const cphvb_index shape[],
                    cphvb_index* dimbound)
{
    dimbound[ndim -1] = shape[ndim -1];
    for (int i = ndim -2 ; i >= 0; --i)
    {
        dimbound[i] = dimbound[i+1] * shape[i];
    }
}


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity.
 */
cphvb_index cphvb_calc_offset(cphvb_intp ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[],
                              const cphvb_index element)
{
    cphvb_index offset = 0;
    cphvb_index dimIndex;
    cphvb_intp i;
    for (i = 0; i < ndim; ++i)
    {
        dimIndex = element % cphvb_nelements(ndim - i, &shape[i]);
        if (i != ndim - 1)
            dimIndex = dimIndex / cphvb_nelements(ndim - (i+1), &shape[i+1]);
        offset += dimIndex * stride[i];
    }
    return offset;
}

/* Find the base array for a given array/view
 *
 * @view   Array/view in question
 * @return The Base array
 */
cphvb_array* cphvb_base_array(const cphvb_array* view)
{
    if(view->base == NULL)
    {
        return (cphvb_array*)view;
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
 * @return Error code (CPHVB_SUCCESS, CPHVB_ERROR)
 */
cphvb_error cphvb_data_set(cphvb_array* array, cphvb_data_ptr data)
{
    cphvb_array* base;

    if(array == NULL)
    {
        fprintf(stderr, "Attempt to set data pointer for a null array\n");
        return CPHVB_ERROR;
    }

    base = cphvb_base_array(array);

    if(base->data != NULL && data != NULL)
    {
        fprintf(stderr, "Attempt to set data pointer an array with existing data pointer\n");
        return CPHVB_ERROR;
    }

    base->data = data;

    return CPHVB_SUCCESS;
}

/* Get the data pointer for the array.
 *
 * @array The array in question
 * @result Output area
 * @return Error code (CPHVB_SUCCESS, CPHVB_ERROR)
 */
cphvb_error cphvb_data_get(cphvb_array* array, cphvb_data_ptr* result)
{
    cphvb_array* base;

    if(array == NULL)
    {
        fprintf(stderr, "Attempt to get data pointer for a null array\n");
        return CPHVB_ERROR;
    }

    base = cphvb_base_array(array);

    *result = base->data;

    return CPHVB_SUCCESS;
}

/* Allocate data memory for the given array if not already allocated.
 * If @array is a view, the data memory for the base array is allocated.
 * NB: It does NOT initiate the memory.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_data_malloc(cphvb_array* array)
{
    cphvb_intp bytes;
    cphvb_array* base;

    if(array == NULL)
        return CPHVB_SUCCESS;

    base = cphvb_base_array(array);

    if(base->data != NULL)
        return CPHVB_SUCCESS;

    bytes = cphvb_array_size(base);
    if(bytes <= 0)
        return CPHVB_SUCCESS;

    base->data = cphvb_memory_malloc(bytes);
    if(base->data == NULL)
    {
        int errsv = errno;//mmap() sets the errno.
        printf("cphvb_data_malloc() could not allocate a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return CPHVB_OUT_OF_MEMORY;
    }

    return CPHVB_SUCCESS;
}

/* Frees data memory for the given array.
 * For convenience array is allowed to be NULL.
 *
 * @array  The array in question
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_data_free(cphvb_array* array)
{
    cphvb_intp bytes;
    cphvb_array* base;

    if(array == NULL)
        return CPHVB_SUCCESS;

    base = cphvb_base_array(array);

    if(base->data == NULL)
        return CPHVB_SUCCESS;

    bytes = cphvb_array_size(base);

    if(cphvb_memory_free(base->data, bytes) != 0)
    {
        int errsv = errno;//munmmap() sets the errno.
        printf("cphvb_data_free() could not free a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return CPHVB_ERROR;
    }
    base->data = NULL;
    return CPHVB_SUCCESS;
}


/* Retrive the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
cphvb_array **cphvb_inst_operands(const cphvb_instruction *instruction)
{
    if (instruction->opcode == CPHVB_USERFUNC)
        return (cphvb_array **) instruction->userfunc->operand;
    else
        return (cphvb_array **) instruction->operand;
}


/* Retrive the operand type of a instruction.
 *
 * @instruction  The instruction in question
 * @operand_no Number of the operand in question
 * @return The operand type
 */
cphvb_type cphvb_type_operand(const cphvb_instruction *instruction,
                              cphvb_intp operand_no)
{
    cphvb_array **operands = cphvb_inst_operands(instruction);
    cphvb_array *operand = operands[operand_no];

    if (cphvb_is_constant(operand))
        return instruction->constant.type;
    else
        return operand->type;
}

/* Determines whether two arrays conflicts.
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool cphvb_array_conflict(const cphvb_array *a, const cphvb_array *b)
{
    cphvb_intp i;
    if (a == NULL || b == NULL)
        return false;

    if(a == b)
        return false;

    if(cphvb_base_array(a) != cphvb_base_array(b))
        return false;

    if(a->ndim != b->ndim)
        return true;

    if(a->start != b->start)
        return true;

    for(i=0; i<a->ndim; ++i)
    {
        if(a->shape[i] != b->shape[i])
            return true;
        if(a->stride[i] != b->stride[i])
            return true;
    }
    return false;
}

/* Determines whether the array is a scalar or a broadcast view of a scalar.
 *
 * @array The array
 * @return The boolean answer
 */
bool cphvb_is_scalar(const cphvb_array* array)
{
    return (cphvb_base_array(array)->ndim == 0) ||
        (cphvb_base_array(array)->ndim == 1 && cphvb_base_array(array)->shape[0] == 1);
}

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
bool cphvb_is_constant(const cphvb_array* o)
{
    return (o == NULL);
}

/* Determines whether the two views are the same
 *
 * @a The first array
 * @b The second array
 * @return The boolean answer
 */
bool cphvb_same_view(const cphvb_array* a, const cphvb_array* b)
{
    if (a == b)
        return true;
    if (cphvb_base_array(a) != cphvb_base_array(b))
        return false;
    if (memcmp(((char*)a)+sizeof(cphvb_array*),
               ((char*)b)+sizeof(cphvb_array*),
               sizeof(cphvb_array)-sizeof(cphvb_array*)-sizeof(cphvb_data_ptr)))
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
bool cphvb_disjoint_views(const cphvb_array *a, const cphvb_array *b)
{
    if (a == NULL || b == NULL) // One is a constant 
        return true;
    if(cphvb_base_array(a) != cphvb_base_array(b)) //different base
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

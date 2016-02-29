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
#include <sstream>

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
        if(view->stride[i] != 0)
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
    assert (ndim > 0);
    bh_index res = 1;
    for (int i = 0; i < ndim; ++i)
    {
        res *= shape[i];
    }
    return res;
}

bh_index bh_nelements(const bh_view& view)
{
    return bh_nelements(view.ndim,view.shape);
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
    return (bh_view *) &instruction->operand;
}

/* Determines whether the base array is a scalar.
 *
 * @view The view
 * @return The boolean answer
 */
bool bh_is_scalar(const bh_view* view)
{
    return bh_nelements(*view) == 1;
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


/* Determines whether instruction 'a' depends on instruction 'b',
 * which is true when:
 *      'b' writes to an array that 'a' access
 *                        or
 *      'a' writes to an array that 'b' access
 *
 * @a The first instruction
 * @b The second instruction
 * @return The boolean answer
 */
bool bh_instr_dependency(const bh_instruction *a, const bh_instruction *b)
{
    const int a_nop = bh_noperands(a->opcode);
    const int b_nop = bh_noperands(b->opcode);
    if(a_nop == 0 or b_nop == 0)
        return false;
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i]))
            return true;
    }
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i]))
            return true;
    }
    return false;
}

/* Determines whether the opcode is a sweep opcode
 * i.e. either a reduction or an accumulate
 *
 * @opcode
 * @return The boolean answer
 */
bool bh_opcode_is_sweep(bh_opcode opcode)
{
    return (bh_opcode_is_reduction(opcode) || bh_opcode_is_accumulate(opcode));
}

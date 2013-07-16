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
#include <bh_compute.h>
#include <assert.h>


template <typename T> bh_error do_nselect(bh_view  *out_index,
                                             bh_view  *out_value,
                                             bh_view  *input,
                                             bh_intp   n,
                                             bh_intp   axis,
                                             bh_opcode opcode)
{

    printf("* do_nselect * \n");
    printf("out_index: ");
    bh_pprint_array(out_index);
    printf("out_value: ");
    bh_pprint_array(out_value);
    printf("input: ");
    bh_pprint_array(input);
    printf("n: %lld\n",(long long int) n);
    printf("axis: %lld\n", (long long int) axis);
    printf("opcode: %s\n", bh_opcode_text(opcode));

    return BH_SUCCESS;
}



/**
 *
 * Implementation of the user-defined funtion "nselect".
 * Note that we follow the function signature defined by bh_userfunc_impl.
 *
 */
bh_error bh_compute_nselect(bh_userfunc *arg, void* ve_arg)
{
    bh_nselect_type *m_arg = (bh_nselect_type *) arg;
    assert(m_arg->nout == 2);
    assert(m_arg->nin == 1);
    bh_view *out_index   = &m_arg->operand[0];
    bh_view *out_value   = &m_arg->operand[1];
    bh_view *input       = &m_arg->operand[2];
    bh_intp n             = m_arg->n;
    bh_intp axis          = m_arg->axis;
    bh_opcode opcode      = m_arg->opcode;

    //Make sure that the arrays memory are allocated.
    if(bh_data_malloc(out_index->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if(bh_data_malloc(out_value->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if(bh_data_malloc(input->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;

    switch (bh_base_array(input)->type)
    {
        case BH_INT8:
            return do_nselect<bh_int8>(out_index, out_value, input, n, axis, opcode);
        case BH_INT16:
            return do_nselect<bh_int16>(out_index, out_value, input, n, axis, opcode);
        case BH_INT32:
            return do_nselect<bh_int32>(out_index, out_value, input, n, axis, opcode);
        case BH_INT64:
            return do_nselect<bh_int64>(out_index, out_value, input, n, axis, opcode);
        case BH_UINT8:
            return do_nselect<bh_uint8>(out_index, out_value, input, n, axis, opcode);
        case BH_UINT16:
            return do_nselect<bh_uint16>(out_index, out_value, input, n, axis, opcode);
        case BH_UINT32:
            return do_nselect<bh_uint32>(out_index, out_value, input, n, axis, opcode);
        case BH_UINT64:
            return do_nselect<bh_uint64>(out_index, out_value, input, n, axis, opcode);
        case BH_FLOAT32:
            return do_nselect<bh_float32>(out_index, out_value, input, n, axis, opcode);
        case BH_FLOAT64:
            return do_nselect<bh_float64>(out_index, out_value, input, n, axis, opcode);
        default:
            return BH_TYPE_NOT_SUPPORTED;
    }
}



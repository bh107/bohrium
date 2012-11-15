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
#include <cphvb_compute.h>
#include <assert.h>


template <typename T> cphvb_error do_nselect(cphvb_array  *out_index, 
                                             cphvb_array  *out_value, 
                                             cphvb_array  *input,
                                             cphvb_intp   n,
                                             cphvb_intp   axis,
                                             cphvb_opcode opcode)
{
   
    printf("* do_nselect * \n");
    printf("out_index: ");
    cphvb_pprint_array(out_index);
    printf("out_value: ");
    cphvb_pprint_array(out_value);
    printf("input: ");
    cphvb_pprint_array(input);
    printf("n: %lld\n", n);
    printf("axis: %lld\n", axis);
    printf("opcode: %s\n", cphvb_opcode_text(opcode));
 
    return CPHVB_SUCCESS;
}



/**
 *
 * Implementation of the user-defined funtion "nselect".
 * Note that we follow the function signature defined by cphvb_userfunc_impl.
 *
 */
cphvb_error cphvb_compute_nselect(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_nselect_type *m_arg = (cphvb_nselect_type *) arg;
    assert(m_arg->nout == 2);
    assert(m_arg->nin == 1);
    cphvb_array *out_index   = m_arg->operand[0];
    cphvb_array *out_value   = m_arg->operand[1];
    cphvb_array *input       = m_arg->operand[2];
    cphvb_intp n             = m_arg->n;
    cphvb_intp axis          = m_arg->axis;
    cphvb_opcode opcode      = m_arg->opcode;

    //Make sure that the arrays memory are allocated.
    if(cphvb_data_malloc(out_index) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 
    if(cphvb_data_malloc(out_value) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 
    if(cphvb_data_malloc(input) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY; 

    switch (input->type)
    {
    	case CPHVB_INT8:
		    return do_nselect<cphvb_int8>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_INT16:
		    return do_nselect<cphvb_int16>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_INT32:
		    return do_nselect<cphvb_int32>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_INT64:
		    return do_nselect<cphvb_int64>(out_index, out_value, input, n, axis, opcode);
        case CPHVB_UINT8:
		    return do_nselect<cphvb_uint8>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_UINT16:
		    return do_nselect<cphvb_uint16>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_UINT32:
	        return do_nselect<cphvb_uint32>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_UINT64:
		    return do_nselect<cphvb_uint64>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_FLOAT32:
		    return do_nselect<cphvb_float32>(out_index, out_value, input, n, axis, opcode);
    	case CPHVB_FLOAT64:
		    return do_nselect<cphvb_float64>(out_index, out_value, input, n, axis, opcode);
        default:
            return CPHVB_TYPE_NOT_SUPPORTED;
	}
}



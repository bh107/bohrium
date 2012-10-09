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

template <typename T, typename Instr>
cphvb_error cphvb_compute_reduce_any( cphvb_array* op_out, cphvb_array* op_in, cphvb_index axis )
{
    Instr opcode_func;                          // Functor-pointer
                                                // Data pointers
    T* data_out = (T*) cphvb_base_array(op_out)->data;
    T* data_in  = (T*) cphvb_base_array(op_in)->data;

    cphvb_index stride      = op_in->stride[axis];
    cphvb_index nelements   = op_in->shape[axis];
    cphvb_index i, i, off0;

    if (op_in->ndim == 1) {                     // 1D special case

        *data_out = *data_in;                   // Initialize pseudo-scalar output
                                                // the value of the first element
                                                // in input.
        for(off0 = op_in->start+1, j=1; j < nelements; j++, off0 += stride ) {
            opcode_func( data_out, data_out, (data_in+off0) );
        }

        return CPHVB_SUCCESS;

    } else {                                    // ND general case

        cphvb_array tmp;                            // Copy the input-array meta-data

        tmp.base    = cphvb_base_array(in);
        tmp.type    = in->type;
        tmp.ndim    = in->ndim-1;
        tmp.start   = in->start;

        for(j=0, i=0; i<in->ndim; ++i) {            // Remove the 'axis' dimension from in
            if(i != a->axis) {
                tmp.shape[j]    = in->shape[i];
                tmp.stride[j]   = in->stride[i];
                ++j;
            }
        }
        tmp.data = in->data;
        
        inst.status = CPHVB_INST_PENDING;           // Copy the first element to the output.
        inst.opcode = CPHVB_IDENTITY;
        inst.operand[0] = op_out;
        inst.operand[1] = &tmp;
        inst.operand[2] = NULL;

        //err = traverse_aa<T, T, Instr>( instr );// execute the pseudo-instruction
        err = cphvb_compute_apply( &instr );
        if (err != CPHVB_SUCCESS) {
            return err;
        }
        tmp.start += stride;

        inst.status = CPHVB_INST_PENDING;           // Reduce over the 'axis' dimension.
        inst.opcode = a->opcode;                    // NB: the first element is already handled.
        inst.operand[0] = out;
        inst.operand[1] = out;
        inst.operand[2] = &tmp;
        
        for(i=1; i<nelements; ++i) {
            //err = traverse_aaa<T, T, T, Instr>( instr );
            err = cphvb_compute_apply( &instr );
            if (err != CPHVB_SUCCESS) {
                return err;
            }
            tmp.start += stride;
        }

        return CPHVB_SUCCESS;
    }

}

cphvb_error cphvb_compute_reduce(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;   // Grab function arguments

    cphvb_opcode opcode = a->opcode;                    // Opcode
    cphvb_index axis    = a->axis;                      // The axis to reduce "around"

    cphvb_array *op_out = a->operand[0];                // Output operand
    cphvb_array *op_in  = a->operand[1];                // Input operand

                                                        //  Sanity checks.
    if (cphvb_operands(opcode) != 3) {
        fprintf(stderr, "ERR: opcode: %d is not a binary ufunc, hence it is not suitable for reduction.\n", opcode);
        return CPHVB_ERROR;
    }

	if (cphvb_base_array(a->operand[1])->data == NULL) {
        fprintf(stderr, "ERR: cphvb_compute_reduce; input-operand ( op[1] ) is null.\n");
        return CPHVB_ERROR;
	}

    if (op_in == op_out) {
        fprintf(stderr, "ERR: cphvb_compute_reduce; input and output types are mixed."
                        "Probable causes include reducing over 'LESS', just dont...\n")
        return CPHVB_ERROR;
    }
    
    if (cphvb_data_malloc(op_out) != CPHVB_SUCCESS) {
        fprintf(stderr, "ERR: cphvb_compute_reduce; No memory for reduction-result.\n")
        return CPHVB_OUT_OF_MEMORY;
    }

    // MEGA SWITCH GOES HERE

    return CPHVB_SUCCESS;
}



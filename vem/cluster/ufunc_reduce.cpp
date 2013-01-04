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

#include <cassert>
#include <cphvb.h>
#include <vector>
#include "array.h"
#include "mapping.h"
#include "pgrid.h"
#include "exec.h"
#include "comm.h"

/* Reduces the input chunk to the output chunk.
 * @ufunc_id The ID of the reduce user-defined function
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @out      The output chunk
 * @in       The input chunk
*/
static cphvb_error reduce_chunk(cphvb_intp ufunc_id, cphvb_opcode opcode, 
                                cphvb_intp axis, 
                                cphvb_array *out, cphvb_array *in)
{
    assert(cphvb_base_array(in)->data != NULL); 
    cphvb_error stat, e;
    cphvb_reduce_type ufunc;
    ufunc.id          = ufunc_id; 
    ufunc.nout        = 1;
    ufunc.nin         = 1;
    ufunc.struct_size = sizeof(cphvb_reduce_type);
    ufunc.operand[0]  = out;
    ufunc.operand[1]  = in;
    ufunc.axis        = axis;
    ufunc.opcode      = opcode;
    if((e = exec_local_inst(CPHVB_USERFUNC, NULL, 
           (cphvb_userfunc*)(&ufunc), &stat)) != CPHVB_SUCCESS)
    {
        fprintf(stderr, "Error in ufunc_reduce() when executing "
        "%s: instruction status: %s\n",cphvb_opcode_text(opcode),
        cphvb_error_text(stat));
        return e;
    }
    return CPHVB_SUCCESS;
}


/* Apply the user-defined function "reduce".
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @operand  The output and input operand (global arrays)
 * @ufunc_id The ID of the reduce user-defined function
 * @return   The instruction status 
*/
cphvb_error ufunc_reduce(cphvb_opcode opcode, cphvb_intp axis, 
                         cphvb_array *operand[], cphvb_intp ufunc_id)
{
    cphvb_error e;
    std::vector<ary_chunk> chunks;

    //For the mapping we have to "broadcast" the 'axis' dimension to an 
    //output array view.
    cphvb_array bcast_output = *operand[0];
    bcast_output.base        = cphvb_base_array(operand[0]);
    bcast_output.ndim        = operand[1]->ndim;

    if(operand[1]->ndim == 1)//"Reducing" to a scalar.
    {
        bcast_output.shape[0] = operand[1]->shape[0];
        bcast_output.stride[0] = 0;
    }
    else
    {
        memcpy(bcast_output.shape, operand[1]->shape, bcast_output.ndim * sizeof(cphvb_intp));

        //Insert a zero-stride into the 'axis' dimension
        for(cphvb_intp i=0; i<axis; ++i)
            bcast_output.stride[i] = operand[0]->stride[i];
        bcast_output.stride[axis] = 0;
        for(cphvb_intp i=axis+1; i<bcast_output.ndim; ++i)
            bcast_output.stride[i] = operand[0]->stride[i-1];
    }

    cphvb_array *operands[] = {&bcast_output, operand[1]};
    if((e = mapping_chunks(2, operands, chunks)) != CPHVB_SUCCESS)
        return e;

    assert(chunks.size() > 0);

    //First we handle all chunks that computes the first row
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
    {
        ary_chunk *out_chunk = &chunks[c];
        ary_chunk *in_chunk  = &chunks[c+1];
        cphvb_array *out     = &out_chunk->ary;
        cphvb_array *in      = &in_chunk->ary;
    
        if(out_chunk->coord[axis] > 0)
            continue;//Not the first row.
        
        //Lets remove the "broadcasted" dimension from the output again
        out->ndim = operand[0]->ndim;
        if(in->ndim == 1)//Reducing to a scalar
        {
            out->shape[0] = 1;
            out->stride[0] = 1;
        }
        else
        {    
            for(cphvb_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }
        }

        //Lets make sure that all processes have the need input data.
        comm_array_data(in_chunk, out_chunk->rank);

        if(pgrid_myrank != out_chunk->rank)
            continue;//We do not own the output chunk
        
        if((e = reduce_chunk(ufunc_id, opcode, axis, out, in)) != CPHVB_SUCCESS)
            return e;
    }

    //Then we handle all the rest.
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
    {
        ary_chunk *out_chunk = &chunks[c];
        ary_chunk *in_chunk  = &chunks[c+1];
        cphvb_array *out     = &out_chunk->ary;
        cphvb_array *in      = &in_chunk->ary;
 
        if(out_chunk->coord[axis] == 0)//The first row
            continue;

        //Lets remove the "broadcasted" dimension from the output again
        out->ndim = operand[0]->ndim;
        if(in->ndim == 1)//Reducing to a scalar
        {
            out->shape[0] = 1;
            out->stride[0] = 1;
        }
        else
        {    
            for(cphvb_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }
        }

        //Lets make sure that all processes have the need input data.
        comm_array_data(in_chunk, out_chunk->rank);

        if(pgrid_myrank != out_chunk->rank)
            continue;//We do not own the output chunk
        
        //We need a tmp output array.
        cphvb_array tmp = *out;
        tmp.base = NULL;
        tmp.data = NULL;
        tmp.start = 0;
        cphvb_set_continuous_stride(&tmp);

        if((e = reduce_chunk(ufunc_id, opcode, axis, &tmp, in)) != CPHVB_SUCCESS)
            return e;
        
        //Finally, we have to "reduce" the local chunks together
        cphvb_error stat;
        cphvb_array *ops[] = {out, out, &tmp};
        if((e = exec_local_inst(opcode, ops, NULL, &stat)) != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error in ufunc_reduce() when executing "
            "%s: instruction status: %s\n",cphvb_opcode_text(opcode),
            cphvb_error_text(stat));
            return e;
        }
    }
    return CPHVB_SUCCESS;
}

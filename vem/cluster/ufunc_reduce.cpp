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


/* Apply the user-defined function "reduce" for a vector input.
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @operand  The output and input operand (global arrays)
 * @ufunc_id The ID of the reduce user-defined function
 * @return   The instruction status 
*/
static cphvb_error reduce_vector(cphvb_opcode opcode, cphvb_intp axis, 
                                 cphvb_array *operand[], cphvb_intp ufunc_id)
{
    assert(operand[1]->ndim == 1);
    assert(axis == 0);
    cphvb_error e;

    //For the mapping we have to "broadcast" the 'axis' dimension to an 
    //output array view.
    cphvb_array bcast_out  = *operand[0];
    bcast_out.base      = cphvb_base_array(operand[0]);
    bcast_out.ndim      = 1;
    bcast_out.shape[0]  = operand[1]->shape[0];
    bcast_out.stride[0] = 0;

    std::vector<ary_chunk> chunks;
    cphvb_array *operands[] = {&bcast_out, operand[1]};
    mapping_chunks(2, operands, chunks);
    assert(chunks.size() > 0);

    //Master-tmp array that the master will reduce in the end.
    cphvb_array mtmp;
    mtmp.base = NULL;
    mtmp.type = operand[1]->type;
    mtmp.ndim = 1;
    mtmp.start = 0;
    mtmp.shape[0] = pgrid_worldsize;//Potential one scalar per process
    mtmp.stride[0] = 1;
    mtmp.data = NULL;
    cphvb_intp mtmp_count=0;//Number of scalars received 

    ary_chunk *out = &chunks[0];//The output chunks are all identical
    out->ary.shape[0] = 1;//Remove the broadcasted dimension
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size(); c += 2)
    {
        ary_chunk *in  = &chunks[c+1];
        if(pgrid_myrank == in->rank)//We own the input chunk
        {
            //Local-tmp array that the process will reduce 
            cphvb_array ltmp;
            ltmp.type = in->ary.type;
            ltmp.ndim = 1;
            ltmp.shape[0] = 1;
            ltmp.stride[0] = 1;
            ltmp.data = NULL;

            if(pgrid_myrank == out->rank)//We also own the output chunk
            {
                //Lets write directly to the master-tmp array
                ltmp.base = &mtmp;
                ltmp.start = mtmp_count;
                if((e = reduce_chunk(ufunc_id, opcode, axis, &ltmp, &in->ary)) 
                   != CPHVB_SUCCESS)
                    return e;
            }
            else
            {
                //Lets write to a tmp array and send it to the master-process
                ltmp.base = NULL;
                ltmp.start = 0;
                if((e = reduce_chunk(ufunc_id, opcode, axis, &ltmp, &in->ary)) 
                   != CPHVB_SUCCESS)
                    return e;

                //Send to output owner's mtmp array
                MPI_Send(ltmp.data, cphvb_type_size(ltmp.type), MPI_BYTE, out->rank, 0, MPI_COMM_WORLD);
            }
        }

        if(pgrid_myrank == out->rank)//We own the output chunk
        {
            if(pgrid_myrank != in->rank)//We don't own the input chunk
            {
                //Recv from input owner's ltmp to the output owner's mtmp array
                if((e = cphvb_data_malloc(&mtmp)) != CPHVB_SUCCESS)
                    return e;
                MPI_Recv(((char*)mtmp.data) + mtmp_count * cphvb_type_size(mtmp.type), 
                         cphvb_type_size(mtmp.type), MPI_BYTE, in->rank, 0, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            ++mtmp_count;//One scalar added to the master-tmp array
        }
        else
        {   //Only one process can own the final output scalar
            assert(mtmp_count == 0);
        }
    }

    //Lets reduce the master-tmp array if we own it
    if(pgrid_myrank == out->rank && mtmp_count > 0)
    {
        assert(mtmp_count <= pgrid_worldsize);
        //Now we know the number of received scalars
        cphvb_array tmp = mtmp;
        tmp.base = &mtmp;
        tmp.shape[0] = mtmp_count;
        reduce_chunk(ufunc_id, opcode, axis, &out->ary, &tmp);
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

    if(operand[1]->ndim == 1)//"Reducing" to a scalar.
        return reduce_vector(opcode, axis, operand, ufunc_id);

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
    mapping_chunks(2, operands, chunks);
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

        //Lets make sure that all processes have the needed input data.
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

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

#include <cassert>
#include <bh.h>
#include <vector>
#include "array.h"
#include "mapping.h"
#include "pgrid.h"
#include "exec.h"
#include "comm.h"
#include "except.h"
#include "batch.h"
#include "tmp.h"

/* Reduces the input chunk to the output chunk.
 * @ufunc_id The ID of the reduce user-defined function
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @out      The output chunk
 * @in       The input chunk
*/
static void reduce_chunk(bh_intp ufunc_id, bh_opcode opcode, 
                         bh_intp axis, 
                         bh_array *out, bh_array *in)
{
    bh_reduce_type *ufunc = (bh_reduce_type*)tmp_get_misc(sizeof(bh_reduce_type));
    ufunc->id          = ufunc_id; 
    ufunc->nout        = 1;
    ufunc->nin         = 1;
    ufunc->struct_size = sizeof(bh_reduce_type);
    ufunc->operand[0]  = out;
    ufunc->operand[1]  = in;
    ufunc->axis        = axis;
    ufunc->opcode      = opcode;
    batch_schedule(BH_USERFUNC, NULL, (bh_userfunc*)(ufunc));
}


/* Apply the user-defined function "reduce" for a vector input.
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @operand  The output and input operand (global arrays)
 * @ufunc_id The ID of the reduce user-defined function
*/
static void reduce_vector(bh_opcode opcode, bh_intp axis, 
                          bh_array *operand[], bh_intp ufunc_id)
{
    assert(operand[1]->ndim == 1);
    assert(axis == 0);

    //For the mapping we have to "broadcast" the 'axis' dimension to an 
    //output array view.
    bh_array bcast_out  = *operand[0];
    bcast_out.base      = bh_base_array(operand[0]);
    bcast_out.ndim      = 1;
    bcast_out.shape[0]  = operand[1]->shape[0];
    bcast_out.stride[0] = 0;

    std::vector<ary_chunk> chunks;
    bh_array *operands[] = {&bcast_out, operand[1]};
    mapping_chunks(2, operands, chunks);
    assert(chunks.size() > 0);

    //Master-tmp array that the master will reduce in the end.
    bh_array *mtmp = tmp_get_ary();
    mtmp->base = NULL;
    mtmp->type = operand[1]->type;
    mtmp->ndim = 1;
    mtmp->start = 0;
    mtmp->shape[0] = pgrid_worldsize;//Potential one scalar per process
    mtmp->stride[0] = 1;
    mtmp->data = NULL;
    bh_intp mtmp_count=0;//Number of scalars received 

    ary_chunk *out = &chunks[0];//The output chunks are all identical
    out->ary->shape[0] = 1;//Remove the broadcasted dimension
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size(); c += 2)
    {
        ary_chunk *in  = &chunks[c+1];
        if(pgrid_myrank == in->rank)//We own the input chunk
        {
            //Local-tmp array that the process will reduce 
            bh_array *ltmp = tmp_get_ary();
            ltmp->type = in->ary->type;
            ltmp->ndim = 1;
            ltmp->shape[0] = 1;
            ltmp->stride[0] = 1;
            ltmp->data = NULL;

            if(pgrid_myrank == out->rank)//We also own the output chunk
            {
                //Lets write directly to the master-tmp array
                ltmp->base = mtmp;
                ltmp->start = mtmp_count;
                reduce_chunk(ufunc_id, opcode, axis, ltmp, in->ary);
            }
            else
            {
                //Lets write to a tmp array and send it to the master-process
                ltmp->base = NULL;
                ltmp->start = 0;
                reduce_chunk(ufunc_id, opcode, axis, ltmp, in->ary);

                //Send to output owner's mtmp array
                batch_schedule(1, out->rank, ltmp); 

                //Lets free the tmp array
                batch_schedule(BH_FREE, ltmp);
            }
            batch_schedule(BH_DISCARD, ltmp);
            if(in->ary->base != NULL)
                batch_schedule(BH_DISCARD, in->ary);
        }

        if(pgrid_myrank == out->rank)//We own the output chunk
        {
            if(pgrid_myrank != in->rank)//We don't own the input chunk
            {
                //Create a tmp view for receiving
                bh_array *recv_view = tmp_get_ary();
                *recv_view = *mtmp;
                recv_view->base = mtmp;
                recv_view->shape[0] = 1;
                recv_view->start = mtmp_count;

                //Recv from input owner's ltmp to the output owner's mtmp array
                batch_schedule(0, in->rank, recv_view);

                //Cleanup
                batch_schedule(BH_DISCARD, recv_view);
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
        bh_array *tmp = tmp_get_ary();
        *tmp = *mtmp;
        tmp->base = mtmp;
        tmp->shape[0] = mtmp_count;
        reduce_chunk(ufunc_id, opcode, axis, out->ary, tmp);
    
        //Lets cleanup
        batch_schedule(BH_DISCARD, tmp);
        batch_schedule(BH_FREE, mtmp);
        batch_schedule(BH_DISCARD, mtmp);
        if(out->ary->base != NULL)
            batch_schedule(BH_DISCARD, out->ary);
    }
}


/* Apply the user-defined function "reduce".
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @operand  The output and input operand (global arrays)
 * @ufunc_id The ID of the reduce user-defined function
 * @return   The instruction status 
*/
bh_error ufunc_reduce(bh_opcode opcode, bh_intp axis, 
                      bh_array *operand[], bh_intp ufunc_id)
{
    std::vector<ary_chunk> chunks;
    try
    {
        if(operand[1]->ndim == 1)//"Reducing" to a scalar.
        {
            reduce_vector(opcode, axis, operand, ufunc_id);
            return BH_SUCCESS;
        }

        //For the mapping we have to "broadcast" the 'axis' dimension to an 
        //output array view.
        bh_array bcast_output = *operand[0];
        bcast_output.base        = bh_base_array(operand[0]);
        bcast_output.ndim        = operand[1]->ndim;
        memcpy(bcast_output.shape, operand[1]->shape, bcast_output.ndim * sizeof(bh_intp));

        //Insert a zero-stride into the 'axis' dimension
        for(bh_intp i=0; i<axis; ++i)
            bcast_output.stride[i] = operand[0]->stride[i];
        bcast_output.stride[axis] = 0;
        for(bh_intp i=axis+1; i<bcast_output.ndim; ++i)
            bcast_output.stride[i] = operand[0]->stride[i-1];

        bh_array *operands[] = {&bcast_output, operand[1]};
        mapping_chunks(2, operands, chunks);
        assert(chunks.size() > 0);

        //First we handle all chunks that computes the first row
        for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
        {
            ary_chunk *out_chunk = &chunks[c];
            ary_chunk *in_chunk  = &chunks[c+1];
            bh_array *out     = out_chunk->ary;
            bh_array *in      = in_chunk->ary;
        
            if(out_chunk->coord[axis] > 0)
                continue;//Not the first row.
            
            //Lets remove the "broadcasted" dimension from the output again
            out->ndim = operand[0]->ndim;
            for(bh_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }

            //And reduce the 'axis' dimension of the input chunk
            bh_array *tmp = tmp_get_ary(); 
            *tmp = *out;
            tmp->base = NULL;
            tmp->data = NULL;
            tmp->start = 0;
            bh_set_contiguous_stride(tmp);
            if(pgrid_myrank == in_chunk->rank)
            {
                reduce_chunk(ufunc_id, opcode, axis, tmp, in);
            }
            if(in->base != NULL)
                batch_schedule(BH_DISCARD, in); 

            //Lets make sure that all processes have the needed input data.
            comm_array_data(tmp, in_chunk->rank, out_chunk->rank);

            if(pgrid_myrank != out_chunk->rank)
                continue;//We do not own the output chunk
            
            //Finally, we have to "reduce" the local chunks together
            bh_array *ops[] = {out, tmp};
            batch_schedule(BH_IDENTITY, ops, NULL);

            //Cleanup
            batch_schedule(BH_FREE, tmp);
            batch_schedule(BH_DISCARD, tmp);
            if(out->base == NULL)
                batch_schedule(BH_FREE, out);
            batch_schedule(BH_DISCARD, out);
        }

        //Then we handle all the rest.
        for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
        {
            ary_chunk *out_chunk = &chunks[c];
            ary_chunk *in_chunk  = &chunks[c+1];
            bh_array *out     = out_chunk->ary;
            bh_array *in      = in_chunk->ary;
     
            if(out_chunk->coord[axis] == 0)
                continue;//The first row

            //Lets remove the "broadcasted" dimension from the output again
            out->ndim = operand[0]->ndim;
            for(bh_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }

            //And reduce the 'axis' dimension of the input chunk
            bh_array *tmp = tmp_get_ary(); 
            *tmp = *out;
            tmp->base = NULL;
            tmp->data = NULL;
            tmp->start = 0;
            bh_set_contiguous_stride(tmp);
            if(pgrid_myrank == in_chunk->rank)
            {
                reduce_chunk(ufunc_id, opcode, axis, tmp, in);
            } 
            if(in->base != NULL)
                batch_schedule(BH_DISCARD, in); 
 
            //Lets make sure that all processes have the needed input data.
            comm_array_data(tmp, in_chunk->rank, out_chunk->rank);

            if(pgrid_myrank != out_chunk->rank)
                continue;//We do not own the output chunk
            
            //Finally, we have to "reduce" the local chunks together
            bh_array *ops[] = {out, out, tmp};
            batch_schedule(opcode, ops, NULL);

            //Cleanup
            batch_schedule(BH_FREE, tmp);
            batch_schedule(BH_DISCARD, tmp);
            if(out->base == NULL)
                batch_schedule(BH_FREE, out);
            batch_schedule(BH_DISCARD, out);
        }
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "[CLUSTER-VEM] Unhandled exception when reducing: \"%s\"", e.what());
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

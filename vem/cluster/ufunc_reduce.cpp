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
 * @opcode   The opcode of the reduce function.
 * @axis     The axis to reduce
 * @out      The output chunk
 * @in       The input chunk
*/
static void reduce_chunk(bh_opcode opcode, bh_intp axis,
                         const bh_view &out, const bh_view &in)
{
    bh_instruction inst;
    inst.opcode = opcode;
    inst.operand[0] = out;
    inst.operand[1] = in;
    inst.constant.value.int64 = axis;
    inst.constant.type = BH_INT64;
    batch_schedule_inst(inst);
}


/* Apply the reduce instruction for a vector input and scalar output.
 * @inst    The reduce instruction.
 * @opcode  The opcode of the reduce function.
*/
static void reduce_vector(bh_instruction *inst, bh_opcode opcode)
{
    assert(inst->operand[1].ndim == 1);
    bh_intp axis = inst->constant.value.int64;
    assert(axis == 0);

    //For the mapping we have to "broadcast" the 'axis' dimension to an
    //output array view.
    bh_view bcast_out  = inst->operand[0];
    bcast_out.base      = bh_base_array(&inst->operand[0]);
    bcast_out.ndim      = 1;
    bcast_out.shape[0]  = inst->operand[1].shape[0];
    bcast_out.stride[0] = 0;

    std::vector<ary_chunk> chunks;
    bh_view operands[] = {bcast_out, inst->operand[1]};
    mapping_chunks(2, operands, chunks);
    assert(chunks.size() > 0);

    //Master-tmp array that the master will reduce in the end.
    bh_base *mtmp = tmp_get_ary(bh_base_array(&inst->operand[1])->type, pgrid_worldsize);
    bh_intp mtmp_count=0;//Number of scalars received

    ary_chunk *out = &chunks[0];//The output chunks are all identical
    out->ary.shape[0] = 1;//Remove the broadcasted dimension
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size(); c += 2)
    {
        ary_chunk *in  = &chunks[c+1];
        if(pgrid_myrank == in->rank)//We own the input chunk
        {
            //Local-tmp array that the process will reduce
            bh_view ltmp;
            ltmp.ndim = 1;
            ltmp.shape[0] = 1;
            ltmp.stride[0] = 1;

            if(pgrid_myrank == out->rank)//We also own the output chunk
            {
                //Lets write directly to the master-tmp array
                ltmp.base = mtmp;
                ltmp.start = mtmp_count;
                reduce_chunk(inst->opcode, axis, ltmp, in->ary);
            }
            else
            {
                //Lets write to a tmp array and send it to the master-process
                ltmp.base = tmp_get_ary(bh_base_array(&in->ary)->type, 1);
                ltmp.start = 0;
                reduce_chunk(inst->opcode, axis, ltmp, in->ary);

                //Send to output owner's mtmp array
                bh_view tmp_view = ltmp;
                tmp_view.base = bh_base_array(&ltmp);
                batch_schedule_comm(1, out->rank, tmp_view);

                //Lets free the tmp array
                batch_schedule_inst(BH_FREE, bh_base_array(&ltmp));
            }
            batch_schedule_inst(BH_DISCARD, bh_base_array(&ltmp));
            if(in->temporary)
                batch_schedule_inst(BH_DISCARD, bh_base_array(&in->ary));
        }

        if(pgrid_myrank == out->rank)//We own the output chunk
        {
            if(pgrid_myrank != in->rank)//We don't own the input chunk
            {
                //Create a tmp view for receiving
                bh_view recv_view;
                recv_view.base = mtmp;
                recv_view.ndim = 1;
                recv_view.shape[0] = 1;
                recv_view.stride[0] = 1;
                recv_view.start = mtmp_count;

                //Recv from input owner's ltmp to the output owner's mtmp array
                batch_schedule_comm(0, in->rank, recv_view);
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
        bh_view tmp;
        tmp.base = mtmp;
        tmp.start = 0;
        tmp.ndim = 1;
        tmp.shape[0] = mtmp_count;
        tmp.stride[0] = 1;
        reduce_chunk(inst->opcode, axis, out->ary, tmp);

        //Lets cleanup
        batch_schedule_inst(BH_FREE, mtmp);
        batch_schedule_inst(BH_DISCARD, mtmp);
        if(out->temporary)
            batch_schedule_inst(BH_DISCARD, bh_base_array(&out->ary));
    }
}


/* Apply the reduce instruction.
 * @inst    The reduce instruction.
 * @opcode  The opcode of the reduce function.
*/
void ufunc_reduce(bh_instruction *inst, bh_opcode opcode)
{
    std::vector<ary_chunk> chunks;
    bh_intp axis = inst->constant.value.int64;

    try
    {
        if(inst->operand[1].ndim == 1)//"Reducing" to a scalar.
            return reduce_vector(inst, opcode);

        //For the mapping we have to "broadcast" the 'axis' dimension to an
        //output array view.
        bh_view bcast_output     = inst->operand[0];
        bcast_output.base        = bh_base_array(&inst->operand[0]);
        bcast_output.ndim        = inst->operand[1].ndim;
        memcpy(bcast_output.shape, inst->operand[1].shape, bcast_output.ndim * sizeof(bh_intp));

        //Insert a zero-stride into the 'axis' dimension
        for(bh_intp i=0; i<axis; ++i)
            bcast_output.stride[i] = inst->operand[0].stride[i];
        bcast_output.stride[axis] = 0;
        for(bh_intp i=axis+1; i<bcast_output.ndim; ++i)
            bcast_output.stride[i] = inst->operand[0].stride[i-1];

        bh_view operands[] = {bcast_output, inst->operand[1]};
        mapping_chunks(2, operands, chunks);
        assert(chunks.size() > 0);

        //First we handle all chunks that computes the first row
        for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
        {
            ary_chunk *out_chunk = &chunks[c];
            ary_chunk *in_chunk  = &chunks[c+1];
            bh_view   *out       = &out_chunk->ary;
            bh_view   *in        = &in_chunk->ary;

            if(out_chunk->coord[axis] > 0)
                continue;//Not the first row.

            //Lets remove the "broadcasted" dimension from the output again
            out->ndim = inst->operand[0].ndim;
            for(bh_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }

            //And reduce the 'axis' dimension of the input chunk
            bh_view tmp = *out;
            tmp.base = tmp_get_ary(bh_base_array(out)->type, bh_nelements(out->ndim, out->shape));
            tmp.start = 0;
            bh_set_contiguous_stride(&tmp);
            if(pgrid_myrank == in_chunk->rank)
            {
                reduce_chunk(inst->opcode, axis, tmp, *in);
            }
            if(in_chunk->temporary)
                batch_schedule_inst(BH_DISCARD, bh_base_array(in));

            //Lets make sure that all processes have the needed input data.
            comm_array_data(tmp, in_chunk->rank, out_chunk->rank);

            if(pgrid_myrank != out_chunk->rank)
                continue;//We do not own the output chunk

            //Finally, we have to "reduce" the local chunks together
            bh_view ops[] = {*out, tmp};
            batch_schedule_inst(BH_IDENTITY, ops, NULL);

            //Cleanup
            batch_schedule_inst(BH_FREE, bh_base_array(&tmp));
            batch_schedule_inst(BH_DISCARD, bh_base_array(&tmp));
            if(out_chunk->temporary)
                batch_schedule_inst(BH_FREE, bh_base_array(out));
            batch_schedule_inst(BH_DISCARD, bh_base_array(out));
        }

        //Then we handle all the rest.
        for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += 2)
        {
            ary_chunk *out_chunk = &chunks[c];
            ary_chunk *in_chunk  = &chunks[c+1];
            bh_view *out         = &out_chunk->ary;
            bh_view *in          = &in_chunk->ary;

            if(out_chunk->coord[axis] == 0)
                continue;//The first row

            //Lets remove the "broadcasted" dimension from the output again
            out->ndim = inst->operand[0].ndim;
            for(bh_intp i=axis; i<out->ndim; ++i)
            {
                out->shape[i] = out->shape[i+1];
                out->stride[i] = out->stride[i+1];
            }

            //And reduce the 'axis' dimension of the input chunk
            bh_view tmp;
            tmp = *out;
            tmp.base = tmp_get_ary(bh_base_array(out)->type, bh_nelements(out->ndim, out->shape));
            tmp.start = 0;
            bh_set_contiguous_stride(&tmp);
            if(pgrid_myrank == in_chunk->rank)
            {
                reduce_chunk(inst->opcode, axis, tmp, *in);
            }
            if(in_chunk->temporary)
                batch_schedule_inst(BH_DISCARD, bh_base_array(in));

            //Lets make sure that all processes have the needed input data.
            comm_array_data(tmp, in_chunk->rank, out_chunk->rank);

            if(pgrid_myrank != out_chunk->rank)
                continue;//We do not own the output chunk

            //Finally, we have to "reduce" the local chunks together
            bh_view ops[] = {*out, *out, tmp};
            batch_schedule_inst(opcode, ops, NULL);

            //Cleanup
            batch_schedule_inst(BH_FREE, bh_base_array(&tmp));
            batch_schedule_inst(BH_DISCARD, bh_base_array(&tmp));
            if(out_chunk->temporary)
                batch_schedule_inst(BH_FREE, bh_base_array(out));
            batch_schedule_inst(BH_DISCARD, bh_base_array(out));
        }
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "[CLUSTER-VEM] Unhandled exception when reducing: \"%s\"", e.what());
    }
}

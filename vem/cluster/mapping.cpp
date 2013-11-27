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
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>
#include "pgrid.h"
#include "array.h"
#include "except.h"
#include "batch.h"
#include "tmp.h"
#include "timing.h"


/* Finds the largest possible array chunk that is only located on one process.
 *
 * @nop         Number of global array operands
 * @operand     List of global array operands
 * @chunks      List of the returned array chunks (output)
 * @start_coord The start coordinate of this chunk search
 * @end_coord   The end coordinate of this chunk search
 * @new_coord   The new coordinate for the next chunk search (output)
 */
static void find_largest_chunk(bh_intp nop,
                               bh_view *operands,
                               std::vector<ary_chunk>& chunks,
                               const bh_intp start_coord[],
                               const bh_intp end_coord[],
                               bh_intp new_coord[])
{
    bh_intp first_chunk = chunks.size();
    bh_intp ndim = operands[0].ndim;
    bh_intp shape[BH_MAXDIM];
    for(bh_intp d=0; d < ndim; ++d)
        shape[d] = std::numeric_limits<bh_intp>::max();

    for(bh_intp o=0; o < nop; ++o)
    {
        const bh_view *ary = &operands[o];
        if(bh_is_constant(ary))
        {
            //Save a dummy chunk
            ary_chunk chunk;
            chunks.push_back(chunk);
            continue;
        }

        //Compute the global start based on the dimension start
        bh_intp start = ary->start;
        for(bh_intp d=0; d < ndim; ++d)
            start += start_coord[d] * ary->stride[d];

        //Compute total array base size
        bh_intp totalsize = bh_base_array(ary)->nelem;

        //Compute local array base size for nrank-1
        bh_intp localsize = totalsize / pgrid_worldsize;

        //Find rank and local start
        int rank;
        if(localsize > 0)
        {
            rank = start / localsize;
            //There may be rank overspill because the local size of
            //the last process may be larger then the 'localsize'
            if(rank > pgrid_worldsize-1)
                rank = pgrid_worldsize-1;
            start -= rank * localsize;
        }
        else
            rank = pgrid_worldsize-1;

        //Convert localsize to be specific for this rank
        if(rank == pgrid_worldsize-1)
           localsize = totalsize / pgrid_worldsize + totalsize % pgrid_worldsize;

        bh_intp offset = 0;
        //Find the largest possible shape
        for(bh_intp d=0; d < ndim; ++d)
        {
            bh_intp max_dim = end_coord[d] - start_coord[d];
            bh_intp dim = max_dim;
            if(ary->stride[d] > 0)
                dim = (bh_intp) ceil((localsize - start - offset) /
                                        (double) ary->stride[d]);
            if(dim > max_dim)
                dim = max_dim;
            if(dim < shape[d])//We only save the smallest shape
                shape[d] = dim;
            offset += (dim-1) * ary->stride[d];
        }
        //Save the chunk
        ary_chunk chunk;
        chunk.rank      = rank;
        chunk.ary.ndim  = ary->ndim;
        memcpy(&chunk.coord, start_coord, ndim * sizeof(bh_intp));
        if(pgrid_myrank == rank)//This is a local array
        {
            chunk.ary.start = start;
            chunk.ary.base = array_get_local(bh_base_array(ary));
            memcpy(chunk.ary.stride, ary->stride, ary->ndim * sizeof(bh_intp));
            chunk.temporary = false;
        }
        else//This is a remote array thus we treat it as a base array.
        {   //Note we will set the stride when we know the final shape
            chunk.ary.start = 0;
        }
        chunks.push_back(chunk);
        assert(0 <= rank && rank < pgrid_worldsize);
    }

    //Save the largest possible shape found to all chunks
    for(bh_intp o=0; o < nop; ++o)
    {
        const bh_view *ary = &operands[o];
        if(bh_is_constant(ary))
            continue;
        ary_chunk *chunk = &chunks[first_chunk+o];

        memcpy(chunk->ary.shape, shape, ndim * sizeof(bh_intp));

        if(chunk->rank != pgrid_myrank)
        {   //Now we know the stride and size of the remote array.
            bh_intp s = 1;
            for(bh_intp i=chunk->ary.ndim-1; i >= 0; --i)
            {
                if(ary->stride[i] == 0)
                    chunk->ary.stride[i] = 0;
                else
                {
                    chunk->ary.stride[i] = s;
                    s *= chunk->ary.shape[i];
                }
            }
            chunk->ary.base  = tmp_get_ary(bh_base_array(ary)->type, s);
            chunk->temporary = true;//This array should be cleared when finished
        }
    }

    //Update coord
    for(bh_intp d=0; d < ndim; ++d)
        new_coord[d] = start_coord[d] + shape[d];

    return;
}

/* Retrieves the set of array chunks that makes up the complete
 * local array-view.
 *
 * @nop         Number of global array operands
 * @operand     List of global array operands
 * @chunks      List of the returned array chunks (output)
 * @start_coord The start coordinate of this chunk search
 * @end_coord   The end coordinate of this chunk search
 */
static void get_chunks(bh_intp nop,
                       bh_view *operands,
                       std::vector<ary_chunk>& chunks,
                       const bh_intp start_coord[],
                       const bh_intp end_coord[])
{
    bh_intp ndim = operands[0].ndim;
    bh_intp new_start_coord[BH_MAXDIM];

    //We are finished when one of the coordinates are out of bound
    for(bh_intp d=0; d < ndim; ++d)
        if(start_coord[d] >= end_coord[d])
            return;

    find_largest_chunk(nop, operands, chunks, start_coord,
                       end_coord, new_start_coord);

    bh_intp corner[BH_MAXDIM];
    memset(corner, 0, ndim * sizeof(bh_intp));
    ++corner[ndim-1];
    while(1)//Lets start a new chunk search at each corner of the current chunk.
    {
        //Find start and end coord based on the current corner
        bh_intp start[BH_MAXDIM];
        bh_intp end[BH_MAXDIM];
        for(bh_intp d=0; d < ndim; ++d)
        {
            if(corner[d] == 1)
            {
                start[d] = new_start_coord[d];
                end[d] = end_coord[d];
            }
            else
            {
                start[d] = start_coord[d];
                end[d] = new_start_coord[d];
            }
        }

        //Goto the next start cood
        get_chunks(nop, operands, chunks, start, end);

        //Go to next corner
        for(bh_intp d=ndim-1; d >= 0; --d)
        {
            if(++corner[d] > 1)
            {
                corner[d] = 0;
                if(d == 0)
                    return;
            }
            else
                break;
        }
    }
    EXCEPT("Coordinate overflow when mapping chunks");
}

/* Creates a list of local array chunks that enables local
 * execution of the instruction
 *
 * @nop         Number of global array operands
 * @operands    List of global array operands
 * @chunks      The output chunks
 */
void mapping_chunks(bh_intp nop,
                    bh_view *operands,
                    std::vector<ary_chunk>& chunks)
{
    bh_uint64 stime = bh_timing();

    bh_intp coord[BH_MAXDIM];
    memset(coord, 0, operands[0].ndim * sizeof(bh_intp));
    get_chunks(nop, operands, chunks, coord, operands[0].shape);

    bh_timing_save(timing_mapping, stime, bh_timing());
}



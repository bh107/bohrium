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

#include <cphvb.h>
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


/* Finds the largest possible array chunk that is only located on one process.
 * 
 * @nop         Number of global array operands
 * @operand     List of global array operands
 * @chunks      List of the returned array chunks (output)
 * @start_coord The start coordinate of this chunk search
 * @end_coord   The end coordinate of this chunk search
 * @new_coord   The new coordinate for the next chunk search (output)
 */
static void find_largest_chunk(cphvb_intp nop,
                                      cphvb_array *operand[], 
                                      std::vector<ary_chunk>& chunks,  
                                      const cphvb_intp start_coord[],
                                      const cphvb_intp end_coord[],
                                      cphvb_intp new_coord[])
{
    cphvb_intp first_chunk = chunks.size();
    cphvb_intp ndim = operand[0]->ndim;
    cphvb_intp shape[CPHVB_MAXDIM];
    for(cphvb_intp d=0; d < ndim; ++d)
        shape[d] = std::numeric_limits<cphvb_intp>::max();

    for(cphvb_intp o=0; o < nop; ++o)
    {
        const cphvb_array *ary = operand[o];
        if(cphvb_is_constant(ary))
        {
            //Save a dummy chunk
            ary_chunk chunk;
            chunks.push_back(chunk);
            continue;
        }

        //Compute the global start based on the dimension start
        cphvb_intp start = ary->start;
        for(cphvb_intp d=0; d < ndim; ++d)
            start += start_coord[d] * ary->stride[d];
     
        //Compute total array base size
        cphvb_intp totalsize = cphvb_nelements(cphvb_base_array(ary)->ndim, 
                                               cphvb_base_array(ary)->shape);

        //Compute local array base size for nrank-1
        cphvb_intp localsize = totalsize / pgrid_worldsize;

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

        cphvb_intp offset = 0;
        //Find the largest possible shape
        for(cphvb_intp d=0; d < ndim; ++d)
        {
            cphvb_intp max_dim = end_coord[d] - start_coord[d];
            cphvb_intp dim = max_dim;
            if(ary->stride[d] > 0)        
                dim = (cphvb_intp) ceil((localsize - start - offset) / 
                                        (double) ary->stride[d]);
            if(dim > max_dim)
                dim = max_dim;
            if(dim < shape[d])//We only save the smallest shape
                shape[d] = dim;
            offset += (dim-1) * ary->stride[d];
        }
        //Save the chunk
        ary_chunk chunk;
        chunk.rank       = rank;
        chunk.ary        = tmp_get_ary();
        chunk.ary->type  = ary->type;
        chunk.ary->ndim  = ary->ndim;
        chunk.ary->data  = NULL;
        memcpy(&chunk.coord, start_coord, ndim * sizeof(cphvb_intp));
        if(pgrid_myrank == rank)//This is a local array
        {
            chunk.ary->start = start;
            chunk.ary->base = array_get_local(cphvb_base_array(ary));
            memcpy(chunk.ary->stride, ary->stride, ary->ndim * sizeof(cphvb_intp));
        }
        else//This is a remote array thus we treat it as a base array.
        {   //Note we will set the stride when we know the final shape
            chunk.ary->base = NULL;
            chunk.ary->start = 0;
        }
        chunks.push_back(chunk);
        assert(0 <= rank && rank < pgrid_worldsize);
    }

    //Save the largest possible shape found to all chunks
    for(cphvb_intp o=0; o < nop; ++o)
    {
        if(cphvb_is_constant(operand[o]))
            continue;
        ary_chunk *chunk = &chunks[first_chunk+o];

        memcpy(chunk->ary->shape, shape, ndim * sizeof(cphvb_intp));
        
        if(chunk->rank != pgrid_myrank)
        {   //Now we know the strides of the remote array.
            cphvb_intp s = 1;
            for(cphvb_intp i=chunk->ary->ndim-1; i >= 0; --i)
            {    
                chunk->ary->stride[i] = s;
                s *= chunk->ary->shape[i];
            }
        }
    }

    //Update coord
    for(cphvb_intp d=0; d < ndim; ++d)
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
static void get_chunks(cphvb_intp nop,
                       cphvb_array *operand[], 
                       std::vector<ary_chunk>& chunks,  
                       const cphvb_intp start_coord[],
                       const cphvb_intp end_coord[])
{
    cphvb_intp ndim = operand[0]->ndim;
    cphvb_intp new_start_coord[CPHVB_MAXDIM];

    //We are finished when one of the coordinates are out of bound
    for(cphvb_intp d=0; d < ndim; ++d)
        if(start_coord[d] >= end_coord[d])
            return;
   
    find_largest_chunk(nop, operand, chunks, start_coord, 
                       end_coord, new_start_coord);

    cphvb_intp corner[CPHVB_MAXDIM];
    memset(corner, 0, ndim * sizeof(cphvb_intp));
    ++corner[ndim-1];
    while(1)//Lets start a new chunk search at each corner of the current chunk.
    {
        //Find start and end coord based on the current corner
        cphvb_intp start[CPHVB_MAXDIM];
        cphvb_intp end[CPHVB_MAXDIM];
        for(cphvb_intp d=0; d < ndim; ++d)
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
        get_chunks(nop, operand, chunks, start, end);

        //Go to next corner
        for(cphvb_intp d=ndim-1; d >= 0; --d)
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
 * @operand     List of global array operands
 * @chunks      The output chunks
 */
void mapping_chunks(cphvb_intp nop,
                           cphvb_array *operand[],
                           std::vector<ary_chunk>& chunks) 
{
    cphvb_intp coord[CPHVB_MAXDIM];
    memset(coord, 0, operand[0]->ndim * sizeof(cphvb_intp));
    get_chunks(nop, operand, chunks, coord, operand[0]->shape);
}



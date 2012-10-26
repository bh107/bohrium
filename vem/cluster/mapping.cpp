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
#include "darray_extension.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>

static cphvb_error find_largest_chunk(int NPROC, 
                                      const cphvb_instruction *inst, 
                                      std::vector<cphvb_array>& chunks,  
                                      std::vector<darray_ext>& chunks_ext,
                                      const cphvb_intp start_coord[],
                                      const cphvb_intp end_coord[],
                                      cphvb_intp new_coord[])
{
    cphvb_intp first_chunk = chunks.size();
    cphvb_intp ndim = inst->operand[0]->ndim;
    cphvb_intp shape[CPHVB_MAXDIM];
    for(cphvb_intp d=0; d < ndim; ++d)
        shape[d] = -1;

    cphvb_intp nop = cphvb_operands_in_instruction(inst);
    for(cphvb_intp o=0; o < nop; ++o)
    {
        const cphvb_array *ary = inst->operand[o];
        if(ary == NULL)//Operand is a constant
        {
            //Save a dummy chunk
            cphvb_array chunk;
            darray_ext chunk_ext;
            chunks.push_back(chunk);
            chunks_ext.push_back(chunk_ext);
            continue;
        }

        //Compute the global offset based on the dimension offset
        cphvb_intp offset = ary->start;
        for(cphvb_intp d=0; d < ndim; ++d)
            offset += start_coord[d] * ary->stride[d];
     
        //Compute total array base size
        cphvb_intp totalsize=1;
        for(cphvb_intp d=0; d<cphvb_base_array(ary)->ndim; ++d)
            totalsize *= cphvb_base_array(ary)->shape[d];

        //Compute local array base size for nrank-1
        cphvb_intp localsize = totalsize / NPROC;
        if(localsize == 0)
            localsize = 1;

        //Find the rank
        cphvb_intp rank = offset / localsize;
        //Convert to local offset
        offset = offset % localsize;
        //Convert localsize to be specific for this rank
        if(rank == NPROC-1)
           localsize = totalsize / NPROC + totalsize % NPROC; 

        //Find the largest possible shape
        for(cphvb_intp d=0; d < ndim; ++d)
        {
            cphvb_intp max_dim = end_coord[d] - start_coord[d];
            cphvb_intp dim = max_dim;
            if(ary->stride[d] > 0)        
                dim = (cphvb_intp) ceil((localsize - offset) / (double) ary->stride[d]);
            if(dim > max_dim)
                dim = max_dim;
            if(shape[d] == -1 || dim < shape[d])//We only save the smallest shape
                shape[d] = dim;
        }
        //Save the chunk
        cphvb_array chunk;
        darray_ext chunk_ext;
        chunk_ext.rank = rank;
        chunk.base     = cphvb_base_array(ary);
        chunk.type     = ary->type;
        chunk.ndim     = ary->ndim;
        chunk.data     = NULL;
        chunk.start    = offset;
        memcpy(chunk.stride, ary->stride, ary->ndim * sizeof(cphvb_intp));
        chunks.push_back(chunk);
        chunks_ext.push_back(chunk_ext);
    }

    //Save the largest possible shape found to all chunks
    for(cphvb_intp d=0; d < ndim; ++d)
    {
        assert(shape[d] > 0);
        for(cphvb_intp o=0; o < nop; ++o)
            chunks[first_chunk+o].shape[d] = shape[d];
    }

    //Update coord
    for(cphvb_intp d=0; d < ndim; ++d)
        new_coord[d] = start_coord[d] + shape[d];

    return CPHVB_SUCCESS;
}

static cphvb_error get_chunks(int NPROC,
                              const cphvb_instruction *inst, 
                              std::vector<cphvb_array>& chunks,  
                              std::vector<darray_ext>& chunks_ext,
                              const cphvb_intp start_coord[],
                              const cphvb_intp end_coord[])
{
    cphvb_intp ndim = inst->operand[0]->ndim;
    cphvb_intp new_start_coord[CPHVB_MAXDIM];
    cphvb_error err;

    //We are finished when one of the coordinates are out of bound
    for(cphvb_intp d=0; d < ndim; ++d)
        if(start_coord[d] >= end_coord[d])
            return CPHVB_SUCCESS;
    
    if((err = find_largest_chunk(NPROC, inst, chunks, chunks_ext, 
              start_coord, end_coord, new_start_coord)) != CPHVB_SUCCESS)
        return err;

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
        if((err = get_chunks(NPROC, inst, chunks, chunks_ext, start, end)) != CPHVB_SUCCESS)
            return err;

        //Go to next corner
        for(cphvb_intp d=ndim-1; d >= 0; --d)
        {
            if(++corner[d] > 1)
            {
                corner[d] = 0;
                if(d == 0)
                    return CPHVB_SUCCESS;
            }
            else 
                break;
        }
    }
    return CPHVB_ERROR;//This shouldn't be possible
}

//The public function
cphvb_error mapping_chunks(int NPROC,
                           const cphvb_instruction *inst, 
                           std::vector<cphvb_array>& chunks,  
                           std::vector<darray_ext>& chunks_ext)
{
    cphvb_intp coord[CPHVB_MAXDIM];
    memset(coord, 0, inst->operand[0]->ndim * sizeof(cphvb_intp));
    return get_chunks(NPROC, inst, chunks, chunks_ext, coord, inst->operand[0]->shape);
}



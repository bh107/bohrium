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

//Finds the largest dimension size possible
cphvb_error find_largest_chunk_dim(cphvb_intp localsize,
                                   cphvb_intp ndim, 
                                   cphvb_intp stride[], 
                                   cphvb_intp offset, 
                                   cphvb_intp max_dims[], 
                                   cphvb_intp d, 
                                   cphvb_intp dims[])
{
    while(1)
    {
        dims[d] = ceil(localsize - offset / (double) stride[d]);
        if(dims[d] > max_dims[d])//Overflow of the max dimension size
        {
            dims[d] = max_dims[d];
            break;
        }
        else if(dims[d] <= 0)//Overflow of the local dimension size
        {
            dims[d] = 1;
            ++d;
            if(d >= ndim)//All dimensions are overflowing
                return CPHVB_ERROR;
        }
        else
            break;
    }
    
    if(d+1 >= ndim)
        return CPHVB_SUCCESS; 

    //Check for overflow
    cphvb_intp end_elem = offset;
    for(cphvb_intp i=0; i<ndim; ++i)
       end_elem += (dims[i]-1) * stride[i];
    if(end_elem >= localsize)
        --dims[d];
    
    if(dims[d] <= 0)//We have to start over with a higher d
    {
        dims[d] = 1;
        return find_largest_chunk_dim(localsize, ndim, stride, offset, max_dims, d+1, dims);
    }
    else
        return CPHVB_SUCCESS;
}

//Get the largest chunk possible 
cphvb_error get_largest_chunk(cphvb_intp localsize, 
                              cphvb_array *ary, 
                              cphvb_intp dim_offset[], 
                              cphvb_array *chunk,
                              darray_ext *dary)
{
    cphvb_error ret;
    //Find the least significant dimension not completely included in the last chuck
    cphvb_intp incomplete_dim = 0;
    for(cphvb_intp d=ary->ndim-1; d >= 0; --d)
    {
        if(dim_offset[d] != 0 && dim_offset[d] != ary->shape[d])
        {
            incomplete_dim = d;
            break;
        }
    }

    //Compute the global offset based on the dimension offset
    cphvb_intp offset = ary->start;
    for(cphvb_intp d=0; d<ary->ndim; ++d)
        offset += dim_offset[d] * ary->stride[d];

    //Find the rank
    int rank = offset / localsize;
    //Convert to local offset
    offset = offset % localsize;

    //Find maximum dimension sizes
    cphvb_intp max_dim[CPHVB_MAXDIM];
    for(cphvb_intp d=0; d<incomplete_dim; ++d)
        max_dim[d] = 1;
    max_dim[incomplete_dim] = ary->shape[incomplete_dim] - dim_offset[incomplete_dim];
    for(cphvb_intp d=incomplete_dim+1; d<ary->ndim; ++d)
        max_dim[d] = ary->shape[d];

    //Find largest possible dimension
    cphvb_intp dim[CPHVB_MAXDIM];
    memcpy(dim, max_dim, ary->ndim * sizeof(cphvb_intp));
    if((ret = find_largest_chunk_dim(localsize,ary->ndim,ary->stride,offset,max_dim,incomplete_dim,dim)) != CPHVB_SUCCESS)
        return ret;
 
    //Save chunk
    dary->rank   = rank;
    chunk->base  = cphvb_base_array(ary);
    chunk->type  = ary->type;
    chunk->ndim  = ary->ndim;
    chunk->data  = NULL;
    memcpy(chunk->stride, ary->stride, ary->ndim * sizeof(cphvb_intp));
    chunk->start = offset; 
    memcpy(chunk->shape, dim, ary->ndim * sizeof(cphvb_intp));
    
    return CPHVB_SUCCESS;
}

cphvb_error local_array(int NPROC, 
                        cphvb_array *ary, 
                        std::vector<cphvb_array> *chunks,  
                        std::vector<darray_ext> *chunks_ext)
{
    cphvb_error ret;
    cphvb_intp dim_offset[CPHVB_MAXDIM];
    
    if(cphvb_is_constant(ary))
    {
        //Inset empty chunk as placeholder
        cphvb_array chunk;
        darray_ext chunk_ext;
        chunks->push_back(chunk);
        chunks_ext->push_back(chunk_ext);
        return CPHVB_SUCCESS;
    }

    memset(dim_offset, 0, ary->ndim * sizeof(cphvb_intp));
    
    //Compute total array base size
    cphvb_intp totalsize=1;
    for(cphvb_intp i=0; i<cphvb_base_array(ary)->ndim; ++i)
        totalsize *= cphvb_base_array(ary)->shape[i];

    //Compute local array base size
    cphvb_intp localsize = totalsize / NPROC;

    //Handle one chunk at a time
    while(1)
    {
        cphvb_array chunk;
        darray_ext chunk_ext;
        if((ret = get_largest_chunk(localsize, ary, dim_offset, &chunk, &chunk_ext)) != CPHVB_SUCCESS)
            return ret;
        chunks->push_back(chunk);
        chunks_ext->push_back(chunk_ext);

        //Find the least significant dimension not completely included in the last chuck
        cphvb_intp incomplete_dim = -1;
        for(cphvb_intp d=ary->ndim-1; d >= 0; --d)
        {   
            if(chunk.shape[d] != ary->shape[d])
            {
                incomplete_dim = d;
                break;
            }
        }
        if(incomplete_dim == -1)
            return CPHVB_SUCCESS;

        //Update the dimension offsets
        for(cphvb_intp d=incomplete_dim; d >= 0; --d)
        {
            dim_offset[d] += chunk.shape[d];
            if(dim_offset[d] >= ary->shape[d])
            {
                if(d == 0)
                    return CPHVB_SUCCESS;
                dim_offset[d] = 0;
            }
            else
                break;
        }
    }
    return CPHVB_SUCCESS;
}


cphvb_error local_arrays(int NPROC, 
                         cphvb_instruction *inst, 
                         std::vector<cphvb_array>& chunks,  
                         std::vector<darray_ext>& chunks_ext)
{
    cphvb_error ret;
    cphvb_intp dim_offset[CPHVB_MAXDIM];
    memset(dim_offset, 0, inst->operand[0]->ndim * sizeof(cphvb_intp));
    
    //Handle one chunk at a time
    while(1)
    {
        std::vector<cphvb_array>::size_type first_chunk = chunks.size();
        for(cphvb_intp o=0; o < cphvb_operands_in_instruction(inst); ++o)
        {
            cphvb_array chunk;
            darray_ext chunk_ext;
            cphvb_array *ary = inst->operand[o];
            assert(ary != NULL);           
 
            //Compute total array base size
            cphvb_intp totalsize=1;
            for(cphvb_intp i=0; i<cphvb_base_array(ary)->ndim; ++i)
                totalsize *= cphvb_base_array(ary)->shape[i];

            //Compute local array base size
            cphvb_intp localsize = totalsize / NPROC;

            if((ret = get_largest_chunk(localsize, ary, dim_offset, &chunk, &chunk_ext)) != CPHVB_SUCCESS)
                return ret;
            chunks.push_back(chunk);
            chunks_ext.push_back(chunk_ext);
        }
        cphvb_intp min_dim_size[CPHVB_MAXDIM];
        memcpy(min_dim_size, &chunks[first_chunk], chunks[first_chunk].ndim * sizeof(cphvb_intp));
        //Find the greates dimensions included by all operands
        for(cphvb_intp o=1; o < cphvb_operands_in_instruction(inst); ++o)
        {
            cphvb_array *ary = &chunks[first_chunk + o];
            for(cphvb_intp i=0; i < ary->ndim; ++i)
            {
                if(ary->shape[i] < min_dim_size[i])
                    min_dim_size[i] = ary->shape[i];
            }
        }
        //Set the new dimension sizes
        for(cphvb_intp o=0; o < cphvb_operands_in_instruction(inst); ++o)
        {
            cphvb_array *ary = inst->operand[o];
            for(cphvb_intp i=0; i < ary->ndim; ++i)
                ary->shape[i] = min_dim_size[i];
        }

        //Find the least significant dimension not completely included in the last chuck
        cphvb_intp incomplete_dim = -1;
        for(cphvb_intp d=inst->operand[0]->ndim-1; d >= 0; --d)
        {   
            if(min_dim_size[d] != inst->operand[0]->shape[d])
            {
                incomplete_dim = d;
                break;
            }
        }
        if(incomplete_dim == -1)
            return CPHVB_SUCCESS;

        //Update the dimension offsets
        for(cphvb_intp d=incomplete_dim; d >= 0; --d)
        {
            dim_offset[d] += min_dim_size[d];
            if(dim_offset[d] >= inst->operand[0]->shape[d])
            {
                if(d == 0)
                    return CPHVB_SUCCESS;
                dim_offset[d] = 0;
            }
            else
                break;
        }            
    }
}




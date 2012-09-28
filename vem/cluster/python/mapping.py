from operator import mul
import math
from array import array

def find_largest_chunk_dim(localsize, stride, offset, max_dims, d, dims):
    #Find a the largest possible dimension size
    while 1:
        dims[d] = int(math.ceil((localsize - offset) / float(stride[d])))
        if dims[d] > max_dims[d]:#Overflow of the max dimension size
            dims[d] = max_dims[d]
            break
        elif dims[d] <= 0:#Overflow of the local dimension size
            dims[d] = 1
            d += 1
            if d >= len(dims):#All dimensions are overflowing
                return False
        else:
            break
    
    if d+1 >= len(dims):#No more dims
        return True    

    end_elem = offset
    for i in xrange(len(dims)):
        end_elem += (dims[i]-1) * stride[i]
    if end_elem >= localsize:#Overflow of last element
        dims[d] -= 1
    
    if dims[d] <= 0:
        dims[d] = 1
        return find_largest_chunk_dim(localsize,stride,offset,max_dims,d+1,dims)
    else:
        return True


def get_largest_chunk(nproc, ary, dim_offset):
    rank = 0
    incomplete_dim = 0
    totalsize = reduce(mul,ary.base.dim)
    localsize = totalsize / nproc

    #Find the least significant dimension not completely included in the last chuck
    for d in xrange(len(ary.dim)-1,-1,-1):
        if dim_offset[d] != 0 and dim_offset[d] != ary.dim[d]:
            incomplete_dim = d
            break
    
    #Compute the global offset based on the dimension offset
    offset = ary.offset
    for d in xrange(len(ary.dim)):
        offset += dim_offset[d] * ary.stride[d]
 
    #Find the rank
    rank = offset / localsize
    #Convert to local offset
    offset = offset % localsize

    #Find maximum dimension sizes
    max_dim = [1]*len(ary.dim)
    max_dim[incomplete_dim] = ary.dim[incomplete_dim] - dim_offset[incomplete_dim]
    for d in xrange(incomplete_dim+1,len(max_dim)):
        max_dim[d] = ary.dim[d]

    assert reduce(mul,max_dim) > 0
    
    #Find largest chunk
    dim = list(max_dim)
    e = find_largest_chunk_dim(localsize, ary.stride,offset,max_dim,incomplete_dim,dim)
    if e == None:
        assert False

    A = array(nproc)
    A.rank = rank
    A.offset = offset
    A.stride = ary.stride
    A.base = ary.base
    A.dim = dim
    A.dim_offset = list(dim_offset)
    return A 



def local_array(nproc, ary):
    ret = []
    dim_offset = [0]*len(ary.dim)
    while True:
        chunk = get_largest_chunk(nproc, ary, list(dim_offset))
        ret.append(chunk)

        #Find the least significant dimension not completely included in the last chuck
        incomplete_dim = -1
        for d in xrange(len(ary.dim)-1,-1,-1):
            if chunk.dim[d] != ary.dim[d]:
                incomplete_dim = d
                break
        if incomplete_dim == -1:
            return ret

        #Update the dimension offsets
        for d in xrange(incomplete_dim,-1,-1):
            dim_offset[d] += chunk.dim[d]
            if dim_offset[d] >= ary.dim[d]:
                dim_offset[d] = 0
                if d == 0:
                    return ret
            else:
                break
    return ret

def local_arrays(nproc, ops):
    dim_offset = [0]*len(ops[0].dim)
    ret = []
    while True:
        #Get largest chunks
        chunk = [None]*len(ops)
        for i in xrange(len(ops)):
            chunk[i] = get_largest_chunk(nproc, ops[i], dim_offset)
        
        #Find the greates dimensions included by all operands
        min_dim_size = list(chunk[0].dim)
        for i in xrange(1,len(ops)):
            for d in xrange(len(chunk[0].dim)):
                if chunk[i].dim[d] < min_dim_size[d]:
                    min_dim_size[d] = chunk[i].dim[d] 

        #Set the new dimension sizes.
        for i in xrange(len(ops)):
            for d in xrange(len(chunk[0].dim)):
                chunk[i].dim[d] = min_dim_size[d] 

        #Find the least significant dimension not completely included in the last chuck
        incomplete_dim = -1
        for d in xrange(len(ops[0].dim)-1,-1,-1):
            if min_dim_size[d] != ops[0].dim[d]:
                incomplete_dim = d
                break

        ret.append(chunk)

        if incomplete_dim == -1:
            return ret
    
        #Update the dimension offsets
        for d in xrange(incomplete_dim,-1,-1):
            dim_offset[d] += min_dim_size[d]
            if dim_offset[d] >= ops[0].dim[d]:
                dim_offset[d] = 0
                if d == 0:
                    return ret
            else:
                break 


from operator import mul
import math
from array import array


def find_largest_chunk(NPROC, nop, operand, chunks, coord, new_coord):
    print "find_largest_chunk: ", coord, new_coord
    first_chunk = len(chunks)
    ndim = len(operand[0].shape)
    shape = [-1]*ndim
    for o in xrange(nop):
        ary = operand[o]
        #Compute the global offset based on the dimension offset
        offset = ary.offset
        for d in xrange(ndim):
            offset += coord[d] * ary.stride[d]
     
        #Compute total array base size
        totalsize=1;
        for d in xrange(len(operand[0].base.shape)):
            totalsize *= ary.base.shape[d]

        #Compute local array base size for nrank-1
        localsize = totalsize / NPROC;
        if localsize == 0:
            localsize = 1 

        #Find the rank
        rank = offset / localsize
        #Convert to local offset
        offset = offset % localsize
        #Convert localsize to be specific for this rank
        if rank == NPROC:
           localsize = totalsize / NPROC + totalsize % NPROC; 

        for d in xrange(ndim):
            dim = int(math.ceil((localsize - offset) / float(ary.stride[d])))
            dim = (localsize - offset) / ary.stride[d]            
            if dim > ary.shape[d]:
                dim = ary.shape[d]
            if shape[d] == -1 or dim < shape[d]:
                shape[d] = dim
        
        A = array(NPROC)
        A.rank = rank
        A.offset = offset
        A.stride = ary.stride
        A.base = ary.base
        A.shape = [0]*ndim
        chunks.append(A)

    #Update shape
    for d in xrange(ndim):
        if shape[d] <= 0:
            shape[d] = 1
        for o in xrange(nop):
            chunks[first_chunk+o].shape[d] = shape[d]

    for o in xrange(nop):
        print chunks[first_chunk+o].pprint()

    #Update coord
    for d in xrange(ndim):
        new_coord[d] = coord[d] + shape[d]



def get_chunks(NPROC, nop, operand, chunks, coord):
    print "get_chunks: ", coord
    ndim = len(operand[0].shape)
    new_coord = [-1]*ndim
    find_largest_chunk(NPROC, nop, operand, chunks, coord, new_coord)
    print "find_largest_chunk return: ", new_coord
    
    for d in xrange(ndim):  
        if new_coord[d] < operand[0].shape[d]:
            next_coord = list(coord)
            next_coord[d] = new_coord[d]
            get_chunks(NPROC, nop, operand, chunks, next_coord)

        

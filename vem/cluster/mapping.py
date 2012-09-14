from operator import mul
import math

class array:
    def pprint(self):
        print "offset: %d"%self.offset
        print "dim:    %s"%self.dim
        print "stride: %s"%self.stride
        print "base dim: %s"%self.base.dim

        totalsize = reduce(mul,base.dim)
        localsize = totalsize / NPROC
        ret = ["_"," "]*totalsize

        coord = [0]*len(self.dim)
        finished = False
        while not finished:
            p = self.offset
            for d in xrange(len(self.dim)):
                p += coord[d] * self.stride[d]
            ret[p*2] = "*"
            #Next coord
            for d in xrange(len(self.dim)):
                coord[d] += 1
                if coord[d] >= self.dim[d]:
                    coord[d] = 0
                    if d == len(self.dim)-1:
                        finished = True
                else:
                    break
        for t in xrange(1,NPROC):
            p = t * localsize * 2
            ret[p-1] = " | " 
        

        out = ""
        for c in ret:
            out += c
        return out


NPROC = 2


def find_largest_chuck(stride, dims, offset, max_dims, d=0):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC

    dims[d] = int(math.ceil((localsize - offset) / float(stride[d])))
    while dims[d] <= 0:#Overflow at most significant dimension
        dims[d] = 1
        d += 1
        if d < len(dims):
            dims[d] = int(math.ceil((localsize - offset) / float(stride[d])))
        else:
            return None    

    if dims[d] > max_dims[d]:#Overflow of global dimension
        dims[d] = max_dims[d]
    
    if d+1 >= len(dims):#No more dims
        return dims    

    end_elem = offset
    for i in xrange(len(dims)):
        end_elem += (dims[i]-1) * stride[i]
    if end_elem >= localsize:#Overflow of last element
        dims[d] -= 1
    
    if dims[d] <= 0:
        dims[d] = 1
        return find_largest_chuck(stride,dims,offset,max_dims,d=d+1)
    else:
        return dims


def local_array(ary, rank=0, offset=-1):
    totalsize = reduce(mul,ary.base.dim)
    localsize = totalsize / NPROC

    #TODO: sort strides and dim

    if offset == -1:
        offset = ary.offset
    
    dim = find_largest_chuck(list(ary.stride),list(ary.dim),offset,list(ary.dim))
    if dim == None:
        return []    

    A = array()
    A.offset = offset
    A.stride = ary.stride
    A.base = ary.base
    A.dim = dim

    if reduce(mul,A.dim) == reduce(mul,ary.dim):
        return [A]

    #Find the least significant dimension not completely included in the last chuck
    incomplete_dim = 0
    for d in xrange(len(ary.dim)-1,-1,-1):
        if A.dim[d] != ary.dim[d]:
            incomplete_dim = d
            break
    
    offset += A.dim[incomplete_dim] * ary.stride[incomplete_dim]
    return [A] + local_array(ary, rank, offset)

def local_array2(ary, pre_rank_last_view):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC
    print pre_rank_last_view
    for d in xrange(len(ary.dim)-1,-1,-1):
        if pre_rank_last_view.dim[d] != ary.dim[d]:#Dimension was not completed by previous rank
            offset = ary.offset
            #for i in xrange(dim
            #dim = find_largest_chuck(list(ary.stride),list(ary.dim),offset,list(ary.dim))

    
    
base = array()
base.offset = 0
base.base = None
base.dim = [36]
base.stride = [1]


A = array()
A.base = base
A.dim = [2,2,2]
A.offset = 0
A.stride = [14,3,1]

print "RANK 0:"
ret = local_array(A)
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()

print "RANK 1:"
ret = local_array2(A,ret[-1])
"""
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()
"""

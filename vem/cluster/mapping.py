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


def find_largest_chuck(ary, dims, offset, d=0):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC

    dims[d] = int(math.ceil((localsize - offset) / float(ary.stride[d])))
    while dims[d] <= 0:#Overflow at most significant dimension
        dims[d] = 1
        d += 1
        if d < len(dims):
            dims[d] = int(math.ceil((localsize - offset) / float(ary.stride[d])))
        else:
            return None    

    if dims[d] > ary.dim[d]:#Overflow of global dimension
        dims[d] = ary.dim[d]
    
    if d+1 >= len(dims):#No more dims
        return dims    

    end_elem = offset
    for i in xrange(len(dims)):
        end_elem += (dims[i]-1) * ary.stride[i]
    if end_elem >= localsize:#Overflow of last element
        dims[d] -= 1
    
    if dims[d] <= 0:
        dims[d] = 1
        return find_largest_chuck(ary,dims,offset,d=d+1)
    else:
        return dims


def local_array(ary, rank=0, offset=-1):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC

    #TODO: sort strides and dim

    if offset == -1:
        offset = ary.offset
    
    dim = find_largest_chuck(ary,list(ary.dim),offset)
    if dim == None:
        return []    

    A = array()
    A.offset = offset
    A.stride = ary.stride
    A.base = ary.base
    A.dim = dim

    if reduce(mul,A.dim) == reduce(mul,ary.dim):
        return [A]

    d = 0
    while d < len(A.dim) and offset + (A.dim[d]-1) * ary.stride[d] >= localsize:
        d += 1
    if d < len(A.dim):
        return [A] + local_array(ary, rank, offset + A.dim[d] * ary.stride[d])
    else:
        return [A]

 
    
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

ret = local_array(A)
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()

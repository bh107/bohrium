from operator import mul
import math

class array:
    def pprint(self):
        print "rank:   %d"%self.rank
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
            ret[2*(p+localsize*self.rank)] = "*"
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


NPROC = 5

def find_largest_chuck(stride, dims, offset, max_dims, d=0):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC
    dims = list(dims)

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
                return None
        else:
            break
    
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



def local_array(ary, dim_offset=None):

    if dim_offset is None:
        dim_offset = [0]*len(ary.dim)
    
    rank = 0
    incomplete_dim = 0
    totalsize = reduce(mul,ary.base.dim)
    localsize = totalsize / NPROC

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
    
    #Find largest chuck
    dim = find_largest_chuck(ary.stride,max_dim,offset,max_dim,incomplete_dim)
    if dim == None:
        assert False

    assert reduce(mul,dim) > 0
 
    A = array()
    A.rank = rank
    A.offset = offset
    A.stride = ary.stride
    A.base = ary.base
    A.dim = dim

    #Find the least significant dimension not completely included in the last chuck
    for d in xrange(len(ary.dim)-1,-1,-1):
        if dim[d] != ary.dim[d]:
            incomplete_dim = d
            break

    #Update the dimension offsets
    for d in xrange(incomplete_dim,-1,-1):
        dim_offset[d] += dim[d]
        if dim_offset[d] >= ary.dim[d]:
            dim_offset[d] = 0
            if d == 0:
                print "EXIT - dim_offset: %s, max_dim: %s, dim: %s"%(dim_offset, max_dim, dim)
                return [A]
        else:
            break 

    print "dim_offset: %s, max_dim: %s, dim: %s, offset: %d"%(dim_offset, max_dim, dim, offset)

    return [A] + local_array(ary, dim_offset)

    
    
base = array()
base.offset = 0
base.base = None
base.dim = [32]
base.stride = [1]

A = array()
A.base = base
A.dim = [4,2,2]
A.offset = 3
A.stride = [6,3,1]


ret = local_array(A)
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()


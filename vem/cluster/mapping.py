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


NPROC = 2

def find_largest_chuck(stride, dims, offset, max_dims, d=0):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC

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
    A.rank = rank
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

def local_array2(ary, pre):
    totalsize = reduce(mul,base.dim)
    localsize = totalsize / NPROC

    ret = []
    incomplete_dim = 0
    for d in xrange(len(ary.dim)-1,-1,-1):
        if pre.dim[d] != ary.dim[d]:
            incomplete_dim = d
            offset = pre.offset + pre.dim[incomplete_dim] * pre.stride[incomplete_dim] - localsize
            
            max_dim = [1]*len(ary.dim)
            max_dim[incomplete_dim] = ary.dim[incomplete_dim] - pre.dim[incomplete_dim]
            for d in xrange(incomplete_dim+1,len(max_dim)):
                max_dim[d] = ary.dim[d]

            dim = find_largest_chuck(list(ary.stride),list(max_dim),offset,max_dim,incomplete_dim)
            print dim, offset
            
            A = array()
            A.rank = 1
            A.offset = offset
            A.stride = ary.stride
            A.base = ary.base
            A.dim = dim
            ret.append(A)

    return ret
    
    
base = array()
base.offset = 0
base.base = None
base.dim = [36]
base.stride = [1]


A = array()
A.base = base
A.dim = [2,2,2]
A.offset = 1
A.stride = [14,3,1]

print "RANK 0:"
ret = local_array(A)
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()

print "RANK 1:"
ret = local_array2(A,ret[-1])
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()

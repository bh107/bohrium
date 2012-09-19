from operator import mul

class array:
    def __init__(self, nproc):
        self.nproc = nproc

    def pprint(self):
        print "rank:       %d"%self.rank
        print "dim_offset: %s"%self.dim_offset
        print "offset:     %d"%self.offset
        print "dim:        %s"%self.dim
        print "stride:     %s"%self.stride
        print "base dim:   %s"%self.base.dim

        totalsize = reduce(mul,self.base.dim)
        localsize = totalsize / self.nproc
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
        for t in xrange(1,self.nproc):
            p = t * localsize * 2
            ret[p-1] = " | " 
        out = ""
        for c in ret:
            out += c
        return out

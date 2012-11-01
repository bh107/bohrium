from operator import mul

class array:
    def __init__(self, nproc):
        self.nproc = nproc

    def pprint(self):
        print "rank:       %d"%self.rank
        print "offset:     %d"%self.offset
        print "shape:      %s"%self.shape
        print "stride:     %s"%self.stride
        print "base shape: %s"%self.base.shape

        totalsize = reduce(mul,self.base.shape)
        localsize = totalsize / self.nproc
        ret = ["_"," "]*totalsize

        coord = [0]*len(self.shape)
        finished = False
        while not finished:
            p = self.offset
            for d in xrange(len(self.shape)):
                p += coord[d] * self.stride[d]
            ret[2*(p+localsize*self.rank)] = "*"
            #Next coord
            for d in xrange(len(self.shape)):
                coord[d] += 1
                if coord[d] >= self.shape[d]:
                    coord[d] = 0
                    if d == len(self.shape)-1:
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

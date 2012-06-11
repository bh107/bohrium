import cphvbnumpy as np
from numpytest import numpytest, NORMAL_TYPES
import random


class test_views(numpytest):
    def gen_shapes(self, max_ndim, max_dim, iters=0, min_ndim=1):
        for ndim in xrange(min_ndim,max_ndim+1):
            shape = [1]*ndim
            if iters:
                yield shape #Min shape
                yield [max_dim]*(ndim) #Max shape
                for _ in xrange(iters):
                    for d in xrange(len(shape)):
                        shape[d] = self.random.randint(1,max_dim)
                    yield shape
            else:       
                finished = False
                while not finished:
                    yield shape
                    #Find next shape
                    d = ndim-1
                    while True:
                        shape[d] += 1
                        if shape[d] > max_dim:
                            shape[d] = 1
                            d -= 1
                            if d < 0:
                                finished = True
                                break
                        else:
                            break

    def gen_views(self, max_ndim, max_dim, iters=0, min_ndim=1):
        for shape in self.gen_shapes(max_ndim, max_dim, iters, min_ndim):
            #Base array
            A = self.array(shape)
            yield (A,"A = self.array(%s)"%(shape))
            #Views with offset per dimension
            for d in xrange(len(shape)):
                if shape[d] > 1:
                    s = "B = A["
                    for _ in xrange(d):
                        s += ":,"
                    s += "1:,"
                    for _ in xrange(len(shape)-(d+1)):
                        s += ":,"
                    s = s[:-1] + "]"
                    exec s 
                    yield (B,s)

            #Views with negative offset per dimension
            for d in xrange(len(shape)):
                if shape[d] > 1:
                    s = "B = A["
                    for _ in xrange(d):
                        s += ":,"
                    s += ":-1,"
                    for _ in xrange(len(shape)-(d+1)):
                        s += ":,"
                    s = s[:-1] + "]"
                    exec s 
                    yield (B,s)
                    
            #Views with steps per dimension
            for d in xrange(len(shape)):
                if shape[d] > 1:
                    s = "B = A["
                    for _ in xrange(d):
                        s += ":,"
                    s += "::2,"
                    for _ in xrange(len(shape)-(d+1)):
                        s += ":,"
                    s = s[:-1] + "]"
                    exec s 
                    yield (B,s)

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = NORMAL_TYPES
        self.size = 100

    def test_flatten(self):
        for (A,cmd) in self.gen_views(3,64,10):
            yield (np.flatten(A),cmd)

    def test_diagonal(self):
        for (A,cmd) in self.gen_views(2,64,10,min_ndim=2):
            yield (np.diagonal(A),cmd)



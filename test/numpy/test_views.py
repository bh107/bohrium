import cphvbnumpy as np
from numpytest import numpytest,gen_views,TYPES
import random


class test_function(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = TYPES.NORMAL

    def test_flatten(self):
        for (A,cmd) in gen_views(self,3,64,10):
            yield (np.flatten(A),"%s; np.flatten(A)"%cmd)

    def test_diagonal(self):
        for (A,cmd) in gen_views(self,2,64,10,min_ndim=2):
            yield (np.diagonal(A),"%s; np.diagonal(A)"%cmd)


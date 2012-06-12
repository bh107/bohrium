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

class test_ufunc(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0
        self.config['dtypes'] = TYPES.NORMAL 

    def test_ufunc(self):
        fun = [np.add,\
            np.subtract,\
            np.divide,\
            np.true_divide,\
            np.floor_divide,\
            np.multiply,\
            np.greater,\
            np.greater_equal,\
            np.less,\
            np.less_equal,\
            np.not_equal,\
            np.equal,\
            np.logical_and,\
            np.logical_or,\
            np.logical_xor,\
            np.logical_not,\
            np.maximum,\
            np.minimum,\
            np.rint,\
            np.sign,\
            np.conj,\
            np.exp,\
            np.exp2,\
            np.log,\
            np.log2,\
            np.log10,\
            np.log1p,\
            np.expm1,\
            np.sqrt,\
            np.square,\
            np.reciprocal,\
            np.ones_like,\
            np.arcsin,\
            np.arccos,\
            np.arctan,\
            np.arctan2,\
            np.hypot,\
            np.sinh,\
            np.cosh,\
            np.tanh,\
            np.arcsinh,\
            np.arccosh,\
            np.arctanh,\
            np.deg2rad,\
            np.rad2deg,\
            np.isfinite,\
            np.isinf,\
            np.isnan,\
            np.signbit,\
            np.floor,\
            np.ceil,\
            np.trunc,\
            np.modf]
        fun = [np.divide] 
        for f in fun:
            for (A,cmd) in gen_views(self,3,64,10):
                B = self.array(A.shape)
                cmd = "%s; B = array(%s); np.%s(A,B)"%(cmd,A.shape,str(f)[8:-2])
                yield (f(A,B),cmd)


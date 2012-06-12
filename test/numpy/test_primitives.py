import cphvbnumpy as np
from numpytest import numpytest,gen_views,TYPES
import random


class test_ufunc(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.config['dtypes'] = TYPES.NORMAL 

    def test_ufunc(self):
        fun = [np.add,\
            np.subtract,\
            np.divide,\
#           np.true_divide,\ #Not Supported by cphVB
#           np.floor_divide,\#Not Supported by cphVB
            np.multiply,\
            np.greater,\
            np.greater_equal,\
            np.less,\
            np.less_equal,\
            np.not_equal,\
            np.equal,\
            np.logical_and,\
            np.logical_or,\
#           np.logical_xor,\ #Error in simple VE
            np.logical_not,\
            np.maximum,\
            np.minimum,\
            np.rint,\
            np.sign,\
#           np.conj,\        #Not Supported by cphVB
            np.log,\
            np.log2,\
            np.log10,\
            np.log1p,\
            np.sqrt,\
            np.square,\
            np.reciprocal,\
            np.hypot,\
#           np.isfinite,\    #Not Supported by cphVB
#           np.isinf,\       #Not Supported by cphVB
#           np.isnan,\       #Not Supported by cphVB
            np.signbit,\
            np.floor,\
            np.ceil,\
            np.trunc,\
            np.negative,\
#           np.modf,\        #Not Supported by cphVB
            ]
        for f in fun:
            A = self.array(100)  
            cmd = "A = array(%s); np.%s(A,A)"%(A.shape,str(f)[8:-2])
            #print cmd
            yield (f(A,A),cmd)


#Only float  
#np.ldexp,\

#Very Small Input
"""
            np.exp,\
            np.exp2,\
            np.expm1,\
"""

#Speciel math
"""
            np.arcsinh,\
            np.arccosh,\
            np.arctanh,\
            np.sin,\
            np.cos,\
            np.tan,\
            np.sinh,\
            np.cosh,\
            np.tanh,\
            np.arcsin,\
            np.arccos,\
            np.arctan,\
            np.arctan2,\
            np.deg2rad,\
            np.rad2deg,\
"""

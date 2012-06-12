import cphvbnumpy as np
from numpytest import numpytest,TYPES

class test_array_create(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = TYPES.NORMAL
        self.size = 100

    def test_zeros(self):
        res = np.zeros(self.size, dtype=self.runtime['dtype'], cphvb=self.runtime['cphvb'])
        cmd = "np.zeros(%d,dtype=%s)"%(self.size,res.dtype)
        return [(res,cmd)]

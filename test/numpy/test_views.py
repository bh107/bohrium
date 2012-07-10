import cphvbnumpy as np
from numpytest import numpytest,gen_views,TYPES


class test_flatten(numpytest):
    
    def init(self):
        for v in gen_views(3,64,6):
            a = {}
            exec v
            yield (a,v)
                
    def test_flatten(self,a):
        cmd = "res = np.flatten(a[0])"
        exec cmd
        return (res,cmd)

class test_diagonal(numpytest):
    
    def init(self):
        for v in gen_views(2,64,12,min_ndim=2):
            a = {}
            exec v
            yield (a,v)

    def test_diagonal(self,a):
        cmd = "res = np.diagonal(a[0])"
        exec cmd
        return (res,cmd)         



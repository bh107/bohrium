import bohrium as np
from numpytest import numpytest,gen_views,TYPES


class test_doubletranspose(numpytest):

    def init(self):
        for v in gen_views(3,64,2):
            a = {}
            exec v
            yield (a,v)

    def test_doubletranspose(self,a):
        cmd = "res = a[0].T + a[0].T"
        exec cmd
        return (res,cmd)

class test_largedim(numpytest):

    def init(self):
        for v in gen_views(7,8,4):
            a = {}
            exec v
            yield (a,v)

    def test_largedim(self,a):
        cmd = "res = a[0] + (a[0] * 4)"
        exec cmd
        return (res,cmd)

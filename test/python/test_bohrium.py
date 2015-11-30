import numpy as np
import bohrium as bh
from numpytest import numpytest,gen_views,TYPES

class test_bohrium(numpytest):

    def init(self):
        for v in gen_views(1,2):
            a = {}
            exec(v)
            yield (a,v)

    def test_tally(self, a):
        cmd = "res = a[0] + a[0]"
        exec(cmd)
        if bh.check(a[0]):
            bh.target.tally()
        return (res,cmd)


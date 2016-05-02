import numpy as np
import bohrium as bh
from numpytest import numpytest,gen_views,TYPES

class test_bohrium(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        
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

    def test_repeat(self, a):
        cmd = "a[0] += 1; a[0] += 1; res = a[0]"
        exec(cmd)
        return (res,cmd)

import numpy as np
import bohrium as bh
from numpytest import numpytest,gen_views,TYPES
import random
import os
import json
import re

def type_bh2numpy(bh_type):
    return "np.%s"%bh_type[3:].lower()


class test_bh_opcodes(numpytest):#Ufuncs directly mappable to Bohrium

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.ops = bh._info.op

    def init(self):
        for op in self.ops.values():
            self.name = op['name']
            self.nops = op['nop']

            if self.name in ["identity"]:
                continue
            for t in op['type_sig']:
                a = {}
                if self.name in ["arcsin","arctanh","arccos"]:
                    high = ",high=False"
                else:
                    high = ",high=True"
                cmd = ""
                for i in xrange(len(t)):
                    cmd += "a[%d] = self.array((10),np.dtype('%s') %s);"%(i, t[i],high)
                exec(cmd)
                yield (a,cmd)

    def test_ufunc(self,a):

        if bh.check(a[0]):
            cmd = "bh.%s("%self.name
        else:
            cmd = "np.%s("%self.name

        if self.name in ["real","imag"]:
            cmd = "a[0] = %sa[1])"%cmd
        else:
            for i in xrange(1,self.nops):
                cmd += "a[%d],"%(i)
            cmd += "a[0])"
        exec(cmd)
        return (a[0],cmd)


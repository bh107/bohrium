import numpy as np
import numpy
from numpytest import numpytest,gen_views,TYPES
import random
import os
import json
import re

def load_opcodes():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    file_path  = os.path.join(script_dir,'..','..','core','codegen','opcodes.json')
    f = open(file_path) 
    ret = json.loads(f.read())
    f.close()
    return ret

def type_cphvb2numpy(cphvb_type):
    return "np.%s"%cphvb_type[6:].lower()


class test_ufunc(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.ops = load_opcodes()

    def init(self):
        for op in self.ops:
            self.name = op['opcode']
            self.nops = op['nop']

            if op['system_opcode'] or self.name in ["CPHVB_IDENTITY"]:
                continue
            for t in op['types']:
                a = {}
                if self.name in ["CPHVB_ARCSIN","CPHVB_ARCTANH","CPHVB_ARCCOS"]:
                    floating = ",floating=True"
                else:
                    floating = "" 
                cmd = ""
                for i in xrange(len(t)):
                    tname = type_cphvb2numpy(t[i])
                    cmd += "a[%d] = self.array((10),%s%s);"%(i,tname,floating)
                exec cmd
                yield (a,cmd)
                
    def test_ufunc(self,a):
        cmd = "%s("%("np.%s"%self.name[6:].lower())
        for i in xrange(1,self.nops):
            cmd += "a[%d],"%(i)
        cmd += "a[0])"
        exec cmd
        return (a[0],cmd)



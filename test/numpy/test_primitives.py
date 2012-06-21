import cphvbnumpy as np
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

 


class test_ufunc(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.ops = load_opcodes()

    def init(self):
        for op in self.ops:
            self.name = op['opcode']
            self.nops = op['nop']
            if op['system_opcode'] or self.name in ["CPHVB_IDENTITY","CPHVB_LDEXP","CPHVB_POWER","CPHVB_ARCSIN","CPHVB_ARCTANH","CPHVB_ARCCOS"]:
                continue
            for t in op['types']:
                a = {}
                cmd = ""
                for i in xrange(len(t)):
                    tname = "np.%s"%t[i][6:].lower() if t[i] != "CPHVB_BOOL" else "bool"
                    cmd += "a[%d] = self.array((10),dtype=%s);"%(i,tname)
                exec cmd
                yield (a,cmd)
                
    def test_ufunc(self,a):
        cmd = "%s("%("np.%s"%self.name[6:].lower())
        for i in xrange(1,self.nops):
            cmd += "a[%d],"%(i)
        cmd += "a[0])"
        exec cmd
        return (a[0],cmd)



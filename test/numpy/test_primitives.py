import cphvbnumpy as np
from numpytest import numpytest,gen_views,TYPES
import random
import os
import json
import re

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def load_opcodes():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    file_path  = os.path.join(script_dir,'..','..','core','codegen','opcodes.json')
    f = open(file_path) 
    ret = json.loads(comment_remover(f.read()))
    f.close()
    return ret

 


class test_ufunc(numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.ops = load_opcodes()
        

    def test_ufunc(self):
        for name in self.ops:
            op = self.ops[name]
            if op['system_opcode'] or name in ["CPHVB_IDENTITY","CPHVB_LDEXP","CPHVB_POWER","CPHVB_ARCSIN"]:#Ignore system opcodes
                continue
            fname = "np.%s"%name[6:].lower()
            for in_type in op['types']:
                for out_type in op['types'][in_type]:
                    in_t  = "np.%s"%in_type[6:].lower() if in_type != "CPHVB_BOOL" else "bool"
                    out_t = "np.%s"%out_type[6:].lower()if out_type != "CPHVB_BOOL" else "bool"
                    cmd = "out = self.array((100),dtype=%s);"%(out_t)
                    for i in xrange(1,op['nop']):
                        cmd  += "in%d = self.array((100),dtype=%s);"%(i,in_t)
                    cmd += "%s("%(fname)
                    for i in xrange(1,op['nop']):
                        cmd += "in%d,"%(i)
                    cmd += "out)"
                    print cmd
                    exec cmd 
                    yield (out,cmd)


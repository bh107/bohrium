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


class test_cphvb_opcodes(numpytest):#Ufuncs directly mappable to cphVB

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
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



def get_type_sig(nop, dtype_in, dtype_out):
    sig = [dtype_out]
    for i in xrange(nop-1):
        sig += [dtype_in]
    return sig 

def type_float(nop):
    sig = []
    for t in ['CPHVB_FLOAT32','CPHVB_FLOAT64']:
        sig += [get_type_sig(nop,t,t)]
    return sig 

def type_int(nop):
    sig = []
    for t in ['CPHVB_INT8','CPHVB_INT16','CPHVB_INT32','CPHVB_INT64']:
        sig += [get_type_sig(nop,t,t)]
    for t in ['CPHVB_UINT8','CPHVB_UINT16','CPHVB_UINT32','CPHVB_UINT64']:
        sig += [get_type_sig(nop,t,t)]
    return sig 

def type_bool(nop):
    return [get_type_sig(nop,'CPHVB_BOOL','CPHVB_BOOL')]

def type_all(nop):
    return type_float(nop) + type_int(nop) + type_bool(nop) 


class test_numpy_ufunc(numpytest):#Ufuncs not directly mappable to cphVB
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.ops = [{'opcode':'floor_divide'},\
                    {'opcode':'true_divide'},\
                    {'opcode':'conjugate'},\
                    {'opcode':'fmod'},\
                    {'opcode':'reciprocal', 'nop':2, 'types':type_int(2)+type_float(2)},\
                    {'opcode':'negative', 'nop':2, 'types':type_all(2)},\
                    {'opcode':'ones_like'},\
                    {'opcode':'_args'},\
                    {'opcode':'fmax'},\
                    {'opcode':'fmin'},\
                    {'opcode':'logaddexp'},\
                    {'opcode':'logaddexp2'},\
                    {'opcode':'degrees'},\
                    {'opcode':'rad2deg'},\
                    {'opcode':'radians'},\
                    {'opcode':'deg2rad'},\
                    {'opcode':'fabs'},\
                    {'opcode':'isnan'},\
                    {'opcode':'isinf'},\
                    {'opcode':'isfinite'},\
                    {'opcode':'copysign'},\
                    {'opcode':'nextafter'},\
                    {'opcode':'spacing'},\
                    {'opcode':'modf'}] 

    def init(self):
        for op in self.ops:
            if op['opcode'] not in ["reciprocal", "negative"]:
                continue

            self.name = op['opcode']
            self.nops = op['nop']
            for t in op['types']:
                a = {}
                cmd = ""
                for i in xrange(len(t)):
                    tname = type_cphvb2numpy(t[i])
                    cmd += "a[%d] = self.array((10),%s);"%(i,tname)
                exec cmd
                yield (a,cmd)
                
    def test_ufunc(self,a):
        cmd = "%s("%("np.%s"%self.name)
        for i in xrange(1,self.nops):
            cmd += "a[%d],"%(i)
        cmd += "a[0])"
        exec cmd
        return (a[0],cmd)




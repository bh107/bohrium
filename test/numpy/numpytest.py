#Test and demonstration of DistNumPy.
import numpy as np
import cphvbnumpy as cnp
import cphvbbridge
import sys
import time
import subprocess
import os
import getopt
from operator import mul
from itertools import izip as zip

ALL_INT      = [np.int8,np.int16,np.int32,np.uint64,np.uint8,np.uint16,np.uint32,np.uint64]
NORMAL_FLOAT = [np.float32,np.float64]
ALL_FLOAT    = [np.float16] + NORMAL_FLOAT
NORMAL_TYPES = ALL_INT + NORMAL_FLOAT
ALL_TYPES    = ALL_INT + ALL_FLOAT

def _array_equal(A,B,maxerror=0.0):
    cphvbbridge.unhandle_array(A)
    cphvbbridge.unhandle_array(B)
    A = A.flatten()
    B = B.flatten()
    if not len(A) == len(B):
        return False

    for i in xrange(len(A)):
        delta = abs(A[i] - B[i])
        if delta > maxerror:
            return delta
    return False

class numpytest:
    def __init__(self):
        self.config = {'maxerror':0.0,'dtypes':[np.float32]}
        self.runtime = {} 
    def array(self,dims,dtype=None,cphvb=None):
        try: 
            total = reduce(mul,dims)
        except TypeError:
            total = dims
        t = dtype if dtype is not None else self.runtime['dtype']
        c = cphvb if cphvb is not None else self.runtime['cphvb']
        res = np.arange(1,total+1,dtype=t).reshape(dims)
        res.cphvb = c
        return res

if __name__ == "__main__":
    pydebug = True
    script_list = []
    exclude_list = []
    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:e:",["file=", "exclude="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ("-f", "--file"):
            script_list.append(a)
        elif o in ("-e", "--exclude"):
            exclude_list.append(a)
        else:
            assert False, "unhandled option"

    if len(script_list) == 0:
        script_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))

    print "*"*11, "Testing the equivalency of cphVB-NumPy and NumPy", "*"*11
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py") and f not in exclude_list:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            #All test classes starts with "test_"
            for cls in [o for o in dir(m) if o.startswith("test_")]:
                cls_obj  = getattr(m, cls)
                cls_inst = cls_obj()
                #All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    print "Testing %s.%s()"%(cls,mth)
                    for t in cls_inst.config['dtypes']:
                        print "\t%s"%(t)
                        cls_inst.runtime['dtype'] = t
                        cls_inst.runtime['cphvb'] = False
                        results1  = getattr(cls_inst,mth)()
                        cls_inst.runtime['cphvb'] = True
                        results2  = getattr(cls_inst,mth)()
                        for (res1,res2) in zip(results1,results2):
                            res = _array_equal(res1, res2, cls_inst.config['maxerror'])
                            if res:
                                print "Delta error:",res
                                sys.exit()

    print "*"*32, "Finish", "*"*32


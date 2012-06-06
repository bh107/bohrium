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
    def __init__(self,cphvb):
        self.config = {'maxerror':0.0,'dtype':[np.float32]}
        self.cphvb  = cphvb
    def array(self,dims,dtype):
        try: 
            total = reduce(mul,dims)
        except TypeError:
            total = dims
        return np.arange(1,total+1,dtype=dtype).reshape(dims)

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
                cls_obj = getattr(m, cls)
                cls1 = cls_obj(False)
                cls2 = cls_obj(True)
                #All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    print "Testing %s.%s()"%(cls,mth)
                    for t in cls1.config['dtypes']:
                        print "\t%s"%(t)
                        results1  = getattr(cls1,mth)({'dtype':t,'cphvb':True})
                        results2  = getattr(cls2,mth)({'dtype':t,'cphvb':False})
                        for (res1,res2) in zip(results1,results2):
                            res = _array_equal(res1, res2, cls1.config['maxerror'])
                            if res:
                                print "Delta error:",res
                                sys.exit()

    print "*"*32, "Finish", "*"*32


#Test and demonstration of DistNumPy.
import numpy as np
import cphvbnumpy as cnp
import cphvbbridge
import sys
import time
import subprocess
import os
import getopt
import random
from operator import mul
from itertools import izip as zip

ALL_INT      = [np.int8,np.int16,np.int32,np.uint64,np.uint8,np.uint16,np.uint32,np.uint64]
NORMAL_FLOAT = [np.float32,np.float64]
ALL_FLOAT    = [np.float16] + NORMAL_FLOAT
NORMAL_TYPES = ALL_INT + NORMAL_FLOAT
ALL_TYPES    = ALL_INT + ALL_FLOAT

class _bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def _array_equal(A,B,maxerror=0.0):
    cphvbbridge.unhandle_array(A)
    cphvbbridge.unhandle_array(B)
    A = A.flatten()
    B = B.flatten()
    if len(A) != len(B):
        return False

    C = np.abs(A - B)
    R = C > maxerror
    if R.any():
        return False

    return True


class numpytest:
    def __init__(self):
        self.config = {'maxerror':0.0,'dtypes':[np.float32]}
        self.runtime = {}
        self.random = random.Random()
        self.random.seed(42)
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

    print "*"*3, "Testing the equivalency of cphVB-NumPy and NumPy", "*"*3
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py") and f not in exclude_list:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            #All test classes starts with "test_"
            for cls in [o for o in dir(m) if o.startswith("test_")]:
                cls_obj  = getattr(m, cls)
                cls_inst1 = cls_obj()
                cls_inst2 = cls_obj()
                cls_inst1.runtime['cphvb'] = False
                cls_inst2.runtime['cphvb'] = True
                #All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    skip_dtypes = False
                    print "Testing %s.%s()"%(cls,mth)
                    for t in cls_inst1.config['dtypes']:
                        print "\t%s"%(t)
                        cls_inst1.runtime['dtype'] = t
                        cls_inst2.runtime['dtype'] = t
                        results1  = getattr(cls_inst1,mth)()
                        results2  = getattr(cls_inst2,mth)()
                        for ((res1,cmd1),(res2,cmd2)) in zip(results1,results2):
                            assert cmd1 == cmd2
                            if not _array_equal(res1, res2, cls_inst1.config['maxerror']):
                                print _bcolors.FAIL   +"[Error] %s.%s (%s)"%(cls,mth,t) + _bcolors.ENDC 
                                print _bcolors.OKBLUE +"[CMD]   %s"%cmd1 + _bcolors.ENDC 
                                #print res1
                                #print res2
                                #sys.exit()
                                skip_dtypes = True
                                break
                        if skip_dtypes:
                            print _bcolors.WARNING + "[Warn]  Skipping the preceding dtypes",
                            print _bcolors.ENDC 
                            break

    print "*"*24, "Finish", "*"*24


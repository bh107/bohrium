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
from itertools import izip as zip

def _array_equal(A,B,maxerror=0.0):
    if type(A) is not type(B):
        return False
    elif (not type(A) == type(np.array([]))) and (not type(A) == type([])):
        if A == B:
            return True
        else:
            return False
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
    return 0.0

use_cphvb = False

class numpytest:
    def __init__(self):
        self.config = {'maxerror':0}
    def array(self,dims):
        cnp.random.seed(42)
        return cnp.random.random(dims,cphvb=use_cphvb) 


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

    print "*"*100
    print "*"*31, "Testing cphVB", "*"*31
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py") and f not in exclude_list:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            cls = m.test_atlas
            print "*"*100
            print "Testing %s (%s)"%(f,cls)

            cls = cls()
            results1  = cls.test_matmul()
            use_cphvb = True
            results2  = cls.test_matmul() 
            
            for (res1,res2) in zip(results1,results2):
                res = _array_equal(res1, res2, cls.config['maxerror'])
                if res != 0.0:
                    print "Delta error:",res

    print "*"*100
    print "*"*46, "Finish", "*"*46
    print "*"*100


#Test and demonstration of the NumPy Bridge.
import numpy as np
import cphvbnumpy as cnp
import cphvbbridge
import sys
import time
import subprocess
import os
import getopt
import random
import warnings
from operator import mul
from itertools import izip as zip

class TYPES:
    NORMAL_INT   = ['np.int32','np.int64','np.uint32','np.uint64']
    ALL_INT      = NORMAL_INT + ['np.int8','np.int16','np.uint8','np.uint16']
    NORMAL_FLOAT = ['np.float32','np.float64']
    ALL_FLOAT    = ['np.float16'] + NORMAL_FLOAT
    NORMAL       = NORMAL_INT + NORMAL_FLOAT
    ALL          = ALL_INT + ALL_FLOAT

class _C:
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
    if type(A) != type(B):
        return False 
    if np.isscalar(A):
        return A == B

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

def gen_shapes(max_ndim, max_dim, iters=0, min_ndim=1):
    for ndim in xrange(min_ndim,max_ndim+1):
        shape = [1]*ndim
        if iters:
            yield shape #Min shape
            yield [max_dim]*(ndim) #Max shape
            for _ in xrange(iters):
                for d in xrange(len(shape)):
                    shape[d] = np.random.randint(1,max_dim)
                yield shape
        else:       
            finished = False
            while not finished:
                yield shape
                #Find next shape
                d = ndim-1
                while True:
                    shape[d] += 1
                    if shape[d] > max_dim:
                        shape[d] = 1
                        d -= 1
                        if d < 0:
                            finished = True
                            break
                    else:
                        break

def gen_views(max_ndim, max_dim, iters=0, min_ndim=1):
    for shape in gen_shapes(max_ndim, max_dim, iters, min_ndim):
        #Base array
        cmd = "a[0] = self.array(%s,np.float32);"%(shape)
        yield cmd
        #Views with offset per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += "1:,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

        #Views with negative offset per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += ":-1,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

        #Views with steps per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += "::2,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

class numpytest:
    def __init__(self):
        self.config = {'maxerror':0.0}
        self.runtime = {}
        self.random = random.Random()
        self.random.seed(42)
    def init(self):
        pass
    def array(self,dims,dtype,floating=False):
        try: 
            total = reduce(mul,dims)
        except TypeError:
            total = dims
            dims = (dims,)
        if dtype is np.bool:
            res = np.random.random_integers(0,1,dims)
        elif floating: 
            res = np.random.random(size=dims)
        else:
            res = np.random.random_integers(1,8,size=dims)
        if len(res.shape) == 0:#Make sure scalars is arrays.
            res = np.asarray(res)
            res.shape = dims
        return np.asarray(res, dtype=dtype)

if __name__ == "__main__":
    warnings.simplefilter('error')#Warnings will raise exceptions
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
                cls_inst = cls_obj()
                #All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    name = "%s/%s/%s"%(f,cls[5:],mth[5:])
                    print "Testing %s"%(name)
                    for (arys,cmd) in getattr(cls_inst,"init")():
                        for a in arys.values():
                            a.cphvb = False
                        (res1,cmd1) = getattr(cls_inst,mth)(arys)
                        res1 = res1.copy()
                        cphvbbridge.flush()
                        for a in arys.values():
                            a.cphvb = True
                        (res2,cmd2) = getattr(cls_inst,mth)(arys)
                        assert cmd1 == cmd2
                        cmd += cmd1
                        try:
                            cphvbbridge.flush() 
                        except RuntimeError as error_msg:
                            print _C.OKBLUE + "[CMD]   %s"%cmd + _C.ENDC
                            print _C.FAIL + str(error_msg) + _C.ENDC 
                        else:
                            if not _array_equal(res1, res2, cls_inst.config['maxerror']):
                                print _C.FAIL + "[Error] %s"%(name) + _C.ENDC 
                                print _C.OKBLUE + "[CMD]   %s"%cmd + _C.ENDC 
                                print _C.OKGREEN + str(res1) + _C.ENDC 
                                print _C.FAIL + str(res2) + _C.ENDC 
                                print 

    print "*"*24, "Finish", "*"*24


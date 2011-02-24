#Test and demonstration of DistNumPy.
import numpy as np
import random
import sys
import time
import subprocess
import os
import getopt

DataSetDir = os.path.join(os.path.join(\
             os.path.dirname(sys.argv[0]), "datasets"), "")

TmpSetDir = os.path.join("/",os.path.join("tmp", ""))

def array_equal(A,B):
    if type(A) is not type(B):
        return False
    elif (not type(A) == type(np.array([]))) and (not type(A) == type([])):
        if A == B:
            return True
        else:
            return False

    A = A.flatten()
    B = B.flatten()
    if not len(A) == len(B):
        return False

    for i in range(len(A)):
        if not A[i] == B[i]:
            return False
    return True

def random_list(dims):
    if len(dims) == 0:
        return random.randint(0,100000)

    list = []
    for i in range(dims[-1]):
        list.append(random_list(dims[0:-1]))
    return list


if __name__ == "__main__":
    pydebug = True
    seed = time.time()
    script_list = []
    exclude_list = []

    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],"s:f:e:",["seed=", "file=", "exclude="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ("-s", "--seed"):
            seed = int(a)
        elif o in ("-f", "--file"):
            script_list.append(a)
        elif o in ("-e", "--exclude"):
            exclude_list.append(a)
        else:
            assert False, "unhandled option"

    if len(script_list) == 0:
        script_list = os.listdir(os.path.dirname(sys.argv[0]))

    random.seed(seed)

    print "*"*100
    print "*"*31, "Testing Distributed Numerical Python", "*"*31
    for i in xrange(len(script_list)):
        f = script_list[i]
        if f.startswith("test_") and f.endswith("py")\
           and f not in exclude_list:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            print "*"*100
            print "Testing %s"%f
            err = False
            msg = ""
            if pydebug:
                r1 = 0; r2 = 0
                r1 = sys.gettotalrefcount()
                try:
                    np.core.multiarray.evalflush()
                    m.run()
                    np.core.multiarray.evalflush()
                except:
                    err = True
                    msg = "Error message: %s"%sys.exc_info()[1]
                r2 = sys.gettotalrefcount()
                if r2 != r1:
                    print "Memory leak - totrefcount: from %d to %d"%(r1,r2)
            else:
                m.run()
            if err:
                print "Error in %s! Random seed: %d"%(f, seed)
                print msg
            else:
                print "Succes"

    print "*"*100
    print "*"*46, "Finish", "*"*46
    print "*"*100


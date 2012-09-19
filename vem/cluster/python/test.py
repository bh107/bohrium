import math
from array import array
from mapping import local_array

NPROC = 2

base = array(NPROC)
base.offset = 0
base.base = None
base.dim = [32]
base.stride = [1]

A = array(NPROC)
A.base = base
A.dim = [4,2,2]
A.offset = 3
A.stride = [8,3,1]


ret = local_array(NPROC,A)
for i in xrange(len(ret)):
    print "chunk:  %d"%i
    print ret[i].pprint()


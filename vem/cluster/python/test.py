import math
from array import array
from mapping import local_arrays
from gen_instructions import gen_instructions


NPROC = 2

base = array(NPROC)
base.offset = 0
base.base = None
base.dim = [28]
base.stride = [1]

A = array(NPROC)
A.base = base
A.dim = [3,3]
A.offset = 5
A.stride = [7,1]

print "ARRAY A"
ret = local_arrays(NPROC,A)
for i in xrange(len(ret)):
    print ret[i].pprint()

B = array(NPROC)
B.base = base
B.dim = [3,3]
B.offset = 4
B.stride = [7,1]

print "ARRAY B"
ret = local_arrays(NPROC,B)
for i in xrange(len(ret)):
    print ret[i].pprint()


gen_instructions([A],[B])


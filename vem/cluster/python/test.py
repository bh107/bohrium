import math
from array import array
from mapping import local_array, local_arrays
from gen_instructions import gen_instructions


NPROC = 4

base = array(NPROC)
base.offset = 0
base.base = None
base.dim = [28]
base.stride = [1]

A = array(NPROC)
A.base = base
A.dim = [2,3]
A.offset = 5
A.stride = [7,1]

print "ARRAY A"
ret = local_array(NPROC,A)
for i in xrange(len(ret)):
    print ret[i].pprint()

B = array(NPROC)
B.base = base
B.dim = [2,3]
B.offset = 4
B.stride = [7,1]

print "ARRAY B"
ret = local_array(NPROC,B)
for i in xrange(len(ret)):
    print ret[i].pprint()

print "\n\nNY\n"
ret = local_arrays(NPROC,[A,B])
for operation in ret:
    print "Operation"
    for op in operation:
        print op.pprint()


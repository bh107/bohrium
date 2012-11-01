import math
from array import array
from mapping import get_chunks


NPROC = 2

base = array(NPROC)
base.offset = 0
base.base = None
base.shape = [28]
base.stride = [1]

A = array(NPROC)
A.base = base
A.shape = [2,2]
A.offset = 0
A.stride = [1,7]

B = array(NPROC)
B.base = base
B.shape = [2,2]
B.offset = 10
B.stride = [7,1]

chunks = []
get_chunks(NPROC,len(A.shape),[A,B], chunks, [0]*len(A.shape))
print "*"*100
for o in xrange(len(chunks)):
    print chunks[o].pprint()


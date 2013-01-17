import numpy as np
import bohriumbridge
import numpytest as t

print "*"*50
A = np.arange(10)
#print A
A.bohrium = True
bohriumbridge.flush()
print "*"*300
C = A + A
print "*"*300
C.bohrium = False

#print C




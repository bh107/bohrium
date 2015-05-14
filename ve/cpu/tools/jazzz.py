#!/usr/bin/env python
import numpy as np

x = np.asarray([1,1,0,-1,-1], dtype=np.complex64)
print(x)
print(np.sign(x))

x = np.asarray([1,1,0,-1,-1])
print(x)
print(np.sign(x))

#sign = np.asarray(x<np.complex64(0.0), dtype=np.complex64)-np.asarray(x>np.complex64(0.0), dtype=np.complex64)
#print(sign)
#print(np.asarray(, dtype=np.complex))

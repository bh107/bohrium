#!/usr/bin/env python
import numpy as np

x = np.asarray([0+1j,0-1j,0+0j,1+0j,1+1j,1-1j,0,1+0j,-1+1j,-1-1j,-1+0j], dtype=np.complex64)
print(x)
print(np.sign(x))

print(np.sign((0+0j)))


#sign = np.asarray(x<np.complex64(0.0), dtype=np.complex64)-np.asarray(x>np.complex64(0.0), dtype=np.complex64)
#print(sign)
#print(np.asarray(, dtype=np.complex))

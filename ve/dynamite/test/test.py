#!/usr/bin/env python
import bohrium as np

a = np.ones((10),dtype=np.float32)
a += 2
b = np.ones((10, 10),dtype=np.float32)
b += 2 
print np.add.reduce(a)
print np.add.reduce(b)


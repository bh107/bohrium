#!/usr/bin/env python
import bohrium as np

a = np.random.random((9,9,9),dtype=np.float32,bohrium=True)
#a = np.ones((9,9,9),dtype=np.float32,bohrium=True)
#a = np.ones((10,10,10),dtype=np.float32,bohrium=True)
#a += 1 

axis = 0
print np.add.reduce(a, axis)



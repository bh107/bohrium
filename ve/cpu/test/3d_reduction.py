#!/usr/bin/env python
import numpy as np

a = np.arange(0,30)
a = np.reshape(a, (5,2,3))

print np.add.reduce(a,0)
print np.add.reduce(a,1)
print np.add.reduce(a,2)

a[0,:,:] = np.add.reduce(a,0)
print a
print "BEH"

import bohrium as np

a = np.arange(0,30)
a = np.reshape(a, (5,2,3))

print np.add.reduce(a,0)
print np.add.reduce(a,1)
print np.add.reduce(a,2)
a[0,:,:] = np.add.reduce(a,0)
print a

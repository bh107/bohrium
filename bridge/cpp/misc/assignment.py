#!/usr/bin/env python
import bohrium as np

for i in xrange(0,10):
    a = np.ones((3,3))
    b = np.ones((3))
    a = b
    np.bridge.flush()
    print "[[[[[ %d ]]]]]" % i

print a

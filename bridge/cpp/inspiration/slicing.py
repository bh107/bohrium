#!/usr/bin/env python
import bohrium as np

print "Slicing Vector!"
a = np.ones((20))
np.bridge.flush()

print "\n\n>>>> vector-neg\n"
a[1:-1:2] = 1.0
np.bridge.flush()

print "\n\n>>>> vector-neg2\n"
a[0:-2:1] = 1.0
np.bridge.flush()

print "\n\n>>>> vector-even\n"
a[0:20:2] = 1.0
np.bridge.flush()

print "\n\n>>>> vector-odd\n"
a[1:20:2] = 1.0
np.bridge.flush()

print "\n\n>>>> vector-sub\n"
a[3:20:2] = 1.0
np.bridge.flush()

print "Slicing Matrix!"
a = np.ones((9,9))
np.bridge.flush()

print "\n\n>>>> matrix-inner-even\n"
a[0:9, 0:9:2] = 1.0
np.bridge.flush()

print "\n\n>>>> matrix-outer-even\n"
a[0:9:2, 0:9] = 1.0
np.bridge.flush()

print "\n\n>>>> matrix-both-even\n"
a[0:9:2, 0:9:2] = 1.0
np.bridge.flush()

print "\n\n>>>> matrix-inner-odd\n"
a[0:9, 3:9:2] = 1.0
np.bridge.flush()

print "\n\n>>>> matrix-outer-odd\n"
a[3:9:2, 0:9] = 1.0
np.bridge.flush()

print "\n\n>>>> matrix-both-odd\n"
a[3:9:2, 3:9:2] = 1.0
np.bridge.flush()

print a[1:9:2, 1:9:2]

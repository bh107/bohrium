#!/usr/bin/env python
import bohrium as np

"""
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

"""

"""
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

"""

"""
print "Assign to slice (MATRIX)"
a = np.ones((9,9))
np.bridge.flush()

print "\n\n>>>> first row\n"
a[0, 0:9:] = 1.0
np.bridge.flush()

print "\n\n>>>> last row\n"
a[-1, 0:9:] = 1.0
np.bridge.flush()

print "\n\n>>>> inbetween row\n"
a[4, 0:9:] = 1.0
np.bridge.flush()

"""

"""
print "Assign to slice2 (MATRIX)"
a = np.ones((9,9))
np.bridge.flush()

print "\n\n>>>> first row\n"
a[0:9:, 0] = 1.0
np.bridge.flush()

print "\n\n>>>> last row\n"
a[0:9:,-1] = 1.0
np.bridge.flush()

print "\n\n>>>> inbetween row\n"
a[0:9:, 4] = 1.0
np.bridge.flush()
"""
print "Assign to slice2 single-element (MATRIX)"
a = np.ones((9,9))
np.bridge.flush()

print "\n\n>>>> first row\n"
a[0, 0] = 3.0
np.bridge.flush()

print "\n\n>>>> inbetween row\n"
a[4, 4] = 3.0
np.bridge.flush()

print "\n\n>>>> last row\n"
a[-1,-1] = 3.0
np.bridge.flush()



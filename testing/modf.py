#!/usr/bin/env python
import numpy as np
import time
import sys
from pprint import pprint as pp

CPHVB=True
#CPHVB=False

a = np.array([ 2.5 ] * 1024, dtype=np.float32, dist=CPHVB)
b = np.array([ 2.5 ] * 1024, dtype=np.float32, dist=CPHVB)

#r1 = np.array([ 5.0 ] * 1024, dtype=np.float32, dist=CPHVB)
#r2 = np.array([ 5.0 ] * 1024, dtype=np.float32, dist=CPHVB)

r1 = np.empty( [1024], dtype=np.float32, dist=CPHVB)
r2 = np.empty( [1024], dtype=np.float32, dist=CPHVB)

np.modf( a, r1, r2 )
print r1[0]
print r2[0]
#print r1
#print r2

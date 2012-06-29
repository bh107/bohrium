import numpy as np
import cphvbbridge
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
iterations = B.size[2]

full = np.empty((H+2,W+2), dtype=np.double, cphvb=B.cphvb)
work = np.empty((H,W), dtype=np.double, cphvb=B.cphvb)
full[:]    = 0.0
full[:,0]  = -273.15
full[:,-1] = -273.15
full[0,:]  =  40.0
full[-1,:] = -273.13

B.start()
for i in xrange(iterations):
  work[:] = full[1:-1, 1:-1]
  work += full[1:-1, 0:-2]
  work += full[1:-1, 2:  ]
  work += full[0:-2, 1:-1]
  work += full[2:  , 1:-1]
  work *= 0.2
  diff = np.absolute(full[1:-1, 1:-1] - work)
  delta = np.add.reduce(diff)
  delta = np.add.reduce(delta)
  full[1:-1, 1:-1] = work
B.stop()

B.pprint()

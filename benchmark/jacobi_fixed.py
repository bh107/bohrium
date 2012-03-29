import numpy as np
import cphvbnumpy
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
iterations = B.size[2]

full = np.empty((H+2,W+2), dtype=B.dtype)
work = np.empty((H,W), dtype=B.dtype)
full[:]    = np.float32(0.0)
full[:,0]  = np.float32(-273.15)
full[:,-1] = np.float32(-273.15)
full[0,:]  =  np.float32(40.0)
full[-1,:] = np.float32(-273.13)

if B.cphvb:
  cphvbnumpy.handle_array(full)
  cphvbnumpy.handle_array(work)

B.start()
for i in xrange(iterations):
  work[:] = full[1:-1, 1:-1]
  work += full[1:-1, 0:-2]
  work += full[1:-1, 2:  ]
  work += full[0:-2, 1:-1]
  work += full[2:  , 1:-1]
  work *= 0.2
  full[1:-1, 1:-1] = work
#  full[1:-1, 1:-1] = (full[1:-1, 1:-1] + full[1:-1, 0:-2] + full[1:-1, 2:  ] +  full[0:-2, 1:-1] + full[2:  , 1:-1]) / 5.0
if B.cphvb:
  cphvbnumpy.unhandle_array(full)

B.stop()

B.pprint()

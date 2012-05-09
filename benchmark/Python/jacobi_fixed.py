import numpy as np
import cphvbbridge
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
iterations = B.size[2]

full = np.empty((H+2,W+2), dtype=B.dtype)
work = np.empty((H,W), dtype=B.dtype)
full[:]    = np.float32(0.0)
full[:,0]  = np.float32(-273.15)  # left column
full[:,-1] = np.float32(-273.15)  # right column
full[0,:]  = np.float32(40.0)    # top row
full[-1,:] = np.float32(-273.13)  # bottom row

center = full[1:-1, 1:-1]
left   = full[1:-1, 0:-2]
right  = full[1:-1, 2:  ]
up     = full[0:-2, 1:-1] 
down   = full[2:  , 1:-1]

if B.cphvb:
  cphvbbridge.handle_array(full)
  cphvbbridge.handle_array(work)

B.start()
for i in xrange(iterations):
  cphvbbridge.flush();
  work[:] = center 
  work += left
  work += right 
  work += up 
  work += down
  work *= 0.2
  center[:] = work

if B.cphvb:
  cphvbbridge.unhandle_array(full)

B.stop()
B.pprint()

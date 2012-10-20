import cphvbnumpy as np
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

full = np.empty((H+2,W+2),  dtype=np.double, cphvb=B.cphvb)
work = np.empty((H,W),      dtype=np.double, cphvb=B.cphvb)

full[:]     = np.float32(0.0)
full[:,0]   = np.float32(-273.15)    # left column
full[:,-1]  = np.float32(-273.15)    # right column
full[0,:]   = np.float32(40.0)       # top row
full[-1,:]  = np.float32(-273.13)    # bottom row

center = full[1:-1, 1:-1]
left   = full[1:-1, 0:-2]
right  = full[1:-1, 2:  ]
up     = full[0:-2, 1:-1] 
down   = full[2:  , 1:-1]

B.start()
for i in xrange(I):
    work[:]   =  center 
    work      += left 
    work      += right 
    work      += up
    work      += down
    work      *= 0.2

    diff  = np.absolute(center - work)
    delta = np.add.reduce(diff)
    delta = np.add.reduce(delta)
    center[:] = work
B.stop()

B.pprint()

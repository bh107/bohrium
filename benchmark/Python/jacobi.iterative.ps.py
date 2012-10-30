import cphvbnumpy as np
import util
import sys

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

full    = np.empty((H+2,W+2),   dtype=np.float64, cphvb=B.cphvb)
work    = np.empty((H+2,W+2),   dtype=np.float64, cphvb=B.cphvb)
full[:] = np.arange((H+2)*(W+2), dtype=np.float64).reshape((H+2,W+2))

B.start()
for i in xrange(I):
    work[1:-1, 1:-1]  = full[1:-1, 1:-1]
    work[1:-1, 1:-1] += full[1:-1, 0:-2]
    work[1:-1, 1:-1] += full[1:-1, 2:  ]
    work[1:-1, 1:-1] += full[0:-2, 1:-1]
    work[1:-1, 1:-1] += full[2:  , 1:-1]
    work[1:-1, 1:-1] *= 0.2
    temp=work; work=full; full=temp

B.stop()
B.pprint()

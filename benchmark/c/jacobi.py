import cphvbnumpy as np
import cphvbbridge as cb
import time
import sys

H = W = int(sys.argv[1])
I = int(sys.argv[2])
cphvb = sys.argv[3].lower() == 'true'

full        = np.empty((H+2,W+2),   dtype=np.float64, cphvb=cphvb)
work        = np.empty((H+2,W+2),   dtype=np.float64, cphvb=cphvb)

full[:]     = np.arange((H+2)*(W+2), dtype=np.float64).reshape((H+2,W+2))
cb.flush()
start=time.time()
for i in xrange(I):
    work[1:-1, 1:-1]  = full[1:-1, 1:-1]
    work[1:-1, 1:-1] += full[1:-1, 0:-2]
    work[1:-1, 1:-1] += full[1:-1, 2:  ] 
    work[1:-1, 1:-1] += full[0:-2, 1:-1]  
    work[1:-1, 1:-1] += full[2:  , 1:-1]
    work[1:-1, 1:-1] *= 0.2
    temp=work; work=full; full=temp

cb.flush()
stop=time.time()
print stop-start

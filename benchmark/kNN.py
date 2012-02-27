import numpy as np
import cphvbnumpy
import util

def compute_targets_pyloop(base, target):
    tmp  = base-target[:,np.newaxis]
    dist = np.add.reduce(tmp)**2
    dist = np.sqrt(dist)
    r  = np.max(dist, axis=0)
    return r

def compute_targets(base, target, tmp1, tmp2):
    base = base[:,np.newaxis]
    target = target[:,:,np.newaxis]
    t = base - target
    #print t.shape, tmp1.shape
    np.subtract(base, target, tmp1)
    tmp1 **= 2
    np.add.reduce(tmp1, out=tmp2)
    np.sqrt(tmp2, tmp2)
    r  = np.max(tmp2, axis=0)
    return r


B = util.Benchmark()
ndims = B.size[0]
db_length = B.size[1]
step = B.size[2]

targets = np.random.random((ndims,db_length), cphvb=B.cphvb)
base    = np.random.random((ndims,db_length), cphvb=B.cphvb)

tmp1 = np.empty((ndims,step,step), dtype=float, dist=B.cphvb)
tmp2 = np.empty((step,step), dtype=float, dist=B.cphvb)

B.start()
for i in xrange(0, db_length, step):
    for j in xrange(0, db_length, step):
        compute_targets(base[:,i:i+step], targets[:,j:j+step], tmp1, tmp2)
B.stop()

B.pprint()


from __future__ import print_function
"""
This benchmark does not seem to make a lot of sense...
"""
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def compute_targets(base, target):
    
    tmp = (base[:,np.newaxis] - target[:,:,np.newaxis])**2
    tmp = np.sum(tmp)
    tmp = np.sqrt(tmp)
    tmp = np.max(tmp, 0)

    return tmp

def main():
    B = util.Benchmark()
    ndims       = B.size[0]
    db_length   = B.size[1]
    I           = B.size[2]

    # Load input-data
    if B.inputfn:
        data    = B.load_arrays(B.inputfn)
        targets = data['targets']
        base    = data['base']
    else:
        targets = np.array(np.random.random((ndims, db_length)), dtype=B.dtype)
        base    = np.array(np.random.random((ndims, db_length)), dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("knn", {'targets': targets, 'base': base})
    
    # Run the Benchmark
    B.start()
    for _ in xrange(I):
        R = compute_targets(base, targets)
    B.stop()
    B.pprint()

    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()

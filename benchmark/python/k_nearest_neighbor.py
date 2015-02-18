from __future__ import print_function
"""
k-Nearest Neighbor
------------------

So what does this code example illustrate?
"""
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def classify(sample, training, group=None, k=1):
    assert sample.ndim == 2
    assert training.ndim == 2
    assert group == None or group.ndim == 1
    assert training.shape[1] == sample.shape[1]
    assert group == None or training.shape[0] == group.size
    assert k > 0 and k < training.shape[0]
    if group == None and k > 1: # k>1 for optimization
        group = np.arange(training.shape[0],dtype=np.int32)
    distance = np.sqrt(np.add.reduce(np.square(sample[:,np.newaxis]-training),-1))
    if k > 1: # for optimization
        groups = np.empty((sample.shape[0],k), dtype=group.dtype, bohrium=group.bohrium)
    for n in xrange(k):
        neighbor_n = np.argmin(distance,-1)
        if k > 1: 
            groups[:,n] = group[neighbor_n]
        elif group == None:  # for optimization
            return neighbor_n
        else:  # for optimization
            return group[neighbor_n]
        if n < k-1:  # for optimization
            neighbor_n += np.arange(distance.shape[0])*distance.shape[1] #convert to a flattened index
            np.flatten(distance)[neighbor_n] = np.inf
    return np.array(map(np.argmax,map(np.bincount,groups)))

# basic version - not cluttered with assertions and optimizations
def classify_basic(sample, training, group=None, k=1):
    if group == None:
        group = np.arange(training.shape[0],dtype=np.int32)
    distance = np.sqrt(np.add.reduce(np.square(sample[:,np.newaxis]-training),-1))
    groups = np.empty((sample.shape[0],k), dtype=group.dtype, bohrium=group.bohrium)
    for n in xrange(k):
        neighbor_n = np.argmin(distance,-1)
        groups[:,n] = group[neighbor_n]
        neighbor_n += np.arange(distance.shape[0])*distance.shape[1] #convert to a flattened index
        np.flatten(distance)[neighbor_n] = np.inf
    return np.array(map(np.argmax,map(np.bincount,groups)))

def main():
    pass    # How would you run this thing!?!?!?!

if __name__ == "__main__":
    main()

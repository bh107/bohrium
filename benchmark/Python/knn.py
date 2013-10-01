import bohrium as np
import util

def compute_targets(base, target):
    
    tmp = (base[:,np.newaxis] - target[:,:,np.newaxis])**2
    tmp = np.sum(tmp)
    tmp = np.sqrt(tmp)
    tmp = np.max(tmp,0)

    return tmp

def main():
    B = util.Benchmark()
    ndims       = B.size[0]
    db_length   = B.size[1]
    i   = B.size[2]

    targets = np.random.random((ndims,db_length), bohrium=B.bohrium)
    base    = np.random.random((ndims,db_length), bohrium=B.bohrium)

    B.start()
    for n in range(0, i):
        compute_targets(base, targets)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()


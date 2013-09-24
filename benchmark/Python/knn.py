import bohrium as np
import util

def compute_targets(base, target):

    base    = base[:,np.newaxis]
    target  = target[:,:,np.newaxis]
    print base.shape, target.shape
    tmp = (base - target)**2
    tmp = np.add.reduce(tmp)

    np.sqrt(tmp, tmp)
    r  = np.max(tmp, axis=0)
    return r

def main():
    B = util.Benchmark()
    ndims       = B.size[0]
    db_length   = B.size[1]

    targets = np.random.random((ndims,db_length), bohrium=B.bohrium)
    base    = np.random.random((ndims,db_length), bohrium=B.bohrium)

    B.start()
    compute_targets(base, targets)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()


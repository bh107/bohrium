from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np
import bohrium.linalg as la

def main():
    B = util.Benchmark()
    N = B.size[0]

    if B.inputfn:
        S = B.load_array()
    else:
        S = B.random_array((N, N))

    if B.dumpinput:
        B.dump_arrays("gauss", {'input':S})

    B.start()
    R = la.gauss(S)
    if util.Benchmark().bohrium:
        R.copy2numpy()

    B.stop()

    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})


if __name__ == "__main__":
    main()

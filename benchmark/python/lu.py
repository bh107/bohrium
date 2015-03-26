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
        a = B.load_array()
    else:
        a = B.random_array((N,N), dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("lu", {'input': a})

    B.start()
    (l, u) = la.lu(a)
    if util.Benchmark().bohrium:
        l.copy2numpy()
        u.copy2numpy()
    B.stop()

    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': u})

if __name__ == "__main__":
    main()

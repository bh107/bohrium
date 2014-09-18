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
        S = np.array(np.random.random((N, N)), dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("gauss", {'input':S})

    B.start()
    R = la.gauss(S)
    B.stop()

    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})


if __name__ == "__main__":
    main()

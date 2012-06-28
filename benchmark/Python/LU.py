#A port from scimark2's LU.c
#But with no pivoting!
#Translated by Brian Vinter

import numpy as np
import util

def main():

    B = util.Benchmark()
    N = B.size[0]
    I = B.size[1]

    A = np.random.random((N,N))
    A.cphvb = B.cphvb

    pivot = np.empty((N), dtype=float)
    pivot.cphvb = B.cphvb
    pivot[:] = 0.0

    ONE = np.empty((1), dtype=float)
    ONE.cphvb = B.cphvb
    ONE[0:-1] = 1.0

    B.start()
    for j in xrange(I):
        if j < N-1:
            recp =  ONE / A[j:j+1,j:j+1]
            A[j:,j:] *= recp[0,:]

        if (j < N-1):
            t1 = A[j+1:,j+1:] - A[j+1:,j] * A[j,j+1:]
            A[j+1:,j+1:] = t1
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()

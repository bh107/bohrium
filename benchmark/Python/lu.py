# A port from scimark2's LU.c
# But with no pivoting!
# Translated by Brian Vinter

import bohrium as np
import util

def main():

    B = util.Benchmark()
    N = B.size[0]
    I = B.size[1]

    A       = np.random.random((N,N),       bohrium=B.bohrium)
    pivot   = np.empty((N), dtype=float,    bohrium=B.bohrium)
    ONE     = np.empty((1), dtype=float,    bohrium=B.bohrium)

    pivot[:]    = 0.0
    ONE[0:-1]   = 1.0

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

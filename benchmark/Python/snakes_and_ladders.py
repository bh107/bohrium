# By  Natalino Busa <https://gist.github.com/natalinobusa/4633275>
from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

"""
#
# These functions are not used unused...
#
def special(p, pos_start, pos_end):
    p[pos_start]= np.zeros(size+1)
    p[pos_start][pos_start]=1

    #make sure that p stays row stocastic
    #i.e the sum of each row must be always kept to 1
    for i in range(size+1):
        pp = p[i][pos_start]
        p[i][pos_start] = 0
        p[i][pos_end]   = p[i][pos_end] + pp

def snake(p, pos_start, pos_end=0):
    if (pos_end<pos_start):
        special(p, pos_start, pos_end)

def ladder(p, pos_start, pos_end=size):
    if (pos_end>pos_start):
        special(p, pos_start, pos_end)
"""

def nullgame(size, dtype):
    p=np.zeros((size+1,size+1), dtype=dtype)

    for i in xrange(size+1):
        for j in xrange(6):
            if (i+j<size):
                p[i][i+j+1]=1.0/6.0

    p[size][size]=1
    p[size-1][size]=6.0/6.0
    p[size-2][size]=5.0/6.0
    p[size-3][size]=4.0/6.0
    p[size-4][size]=3.0/6.0
    p[size-5][size]=2.0/6.0
    p[size-6][size]=1.0/6.0

    return p

def main():

    B           = util.Benchmark()
    size        = B.size[0]
    iterations  = B.size[1]

    if B.visualize:
        from matplotlib import pyplot

    if B.inputfn:
        arrays = B.load_arrays(B.inputfn)
        a = arrays['a']
        p = arrays['p']
    else:
        a = np.array(np.zeros(size+1, dtype=B.dtype))
        p = np.array(nullgame(size,   dtype=B.dtype))

    if B.dumpinput:
        B.dump_arrays("snakes_and_ladders", {"a": a, "p": p})

    m = p   # Initial matrix is p
    pr_end = np.array(np.zeros(iterations, dtype=B.dtype))

    B.start()
    for k in xrange(iterations):
        if B.visualize:
            # Plot the probability distribution at the k-th iteration
            pyplot.figure(1)
            pyplot.plot(m[0][0:size])

        # Store the probability of ending after the k-th iteration
        pr_end[k] = m[0][size]

        # Store/plot the accumulated marginal probability at the k-th iteration
        a = a + m[0]
        if B.visualize:
            pyplot.figure(2)
            pyplot.plot(a[0:size])

        #calculate the stocastic matrix for iteration k+1
        if B.bohrium and B.no_extmethods:
            m = np.array(np.dot(m.copy2numpy(),p.copy2numpy()))
        else:
            m = np.dot(m, p)

    B.stop()
    B.pprint()

    #plot the probability of ending the game
    # after k iterations
    if B.visualize:
        pyplot.figure(3)
        pyplot.plot(pr_end[0:iterations-1])

        #show the three graphs
        pyplot.show()

    if B.verbose:
        print(pr_end)

    if B.outputfn:
        B.tofile(B.outputfn, {'res': pr_end})

if __name__ == "__main__":
    main()

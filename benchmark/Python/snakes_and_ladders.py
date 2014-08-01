# By  Natalino Busa <https://gist.github.com/natalinobusa/4633275>

import util
import bohrium as bh
import numpy as np
from matplotlib import pyplot

B = util.Benchmark()
size=B.size[0]
iterations = B.size[1]

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

def nullgame():
    p=np.zeros((size+1,size+1), dtype=B.dtype)

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

a=bh.array(np.zeros(size+1, dtype=B.dtype), bohrium=B.bohrium)
p=bh.array(nullgame(), bohrium=B.bohrium)

#initial matrix is p
m=p

pr_end=bh.array(np.zeros(iterations,dtype=B.dtype), bohrium=B.bohrium)

B.start()
for k in xrange(iterations):
    if B.visualize:
        #plot the probability distribution at the k-th iteration
        pyplot.figure(1)
        pyplot.plot(m[0][0:size])

    #store the probability of ending after the k-th iteration
    pr_end[k] = m[0][size]

    #store/plot the accumulated marginal probability at the k-th iteration
    a=a+m[0]
    if B.visualize:
        pyplot.figure(2)
        pyplot.plot(a[0:size])

    #calculate the stocastic matrix for iteration k+1
    if B.bohrium and B.no_extmethods:
        m=bh.array(np.dot(m.copy2numpy(),p.copy2numpy()),bohrium=True)
    else:
        m=bh.dot(m,p)
B.stop()
B.pprint()

#plot the probability of ending the game
# after k iterations
if B.visualize:
    pyplot.figure(3)
    pyplot.plot(pr_end[0:iterations-1])

    #show the three graphs
    pyplot.show()


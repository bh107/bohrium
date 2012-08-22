"""
Game of Life
------------

So what does this code example illustrate?
"""
import cphvbnumpy as np

SURVIVE_LOW= 2
SURVIVE_HIGH = 3
SPAWN = 3

def randomstate(height, width, prob=0.2, dtype=np.int32, cphvb=True):
    state = np.zeros((height+2,width+2), dtype=dtype, cphvb=cphvb)
    state[1:-1,1:-1] = np.random.random((width,height), dtype=np.float32, cphvb=cphvb) < prob
    return state

def play(state, iterations):

    cells = state[1:-1,1:-1]
    ul = state[0:-2, 0:-2]
    um = state[0:-2, 1:-1]
    ur = state[0:-2, 2:  ]
    ml = state[1:-1, 0:-2]
    mr = state[1:-1, 2:  ]
    ll = state[2:  , 0:-2]
    lm = state[2:  , 1:-1]
    lr = state[2:  , 2:  ]

    for i in xrange(iterations):
        
        neighbors = ul + um + ur + ml + mr + ll + lm + lr       # count neighbors
        live = neighbors * cells                                # extract live cells neighbors
        stay = (live >= SURVIVE_LOW) & (live <= SURVIVE_HIGH)   # find cells the stay alive
        dead = neighbors * (cells == 0)                         # extract dead cell neighbors
        spawn = dead == SPAWN                                   # find cells that spaw new life
        cells[:] = stay | spawn                                 # save result for next iteration

    return state

if __name__ == "__main__":

    w = 10
    h = 10
    i = 50

    samples = 0
    while (True):

        samples += 1

        #s = randomstate(w, h,cphvb=False)
        #s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32, cphvb=False)

        #s = np.zeros((w,h), dtype=np.int32, cphvb=False)
        s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32, cphvb=False)
        n = s.copy()
        n.cphvb = True

        play( s, 50 )
        play( n, 50 )

        print s
        print n
        break

        bad = False
        for i in xrange(0, w):
            for j in xrange(0, h):
                if not s[i][j] == n[i][j]:
                    print "BAD: ", s[i][j], n[i][j]
                    bad = True

        if bad:
            print s
            break

        if samples % 100 == 0:
            print "Samples checked", samples

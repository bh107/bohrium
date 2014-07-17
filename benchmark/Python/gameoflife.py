"""
Game of Life
------------

So what does this code example illustrate?
"""
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

SURVIVE_LOW     = 2
SURVIVE_HIGH    = 3
SPAWN           = 3

def randomstate(height, width, prob=0.2, dtype=np.int64):
    state               = np.zeros((height+2,width+2), dtype=dtype)
    state[1:-1,1:-1]    = np.array(np.random.random((width,height)), dtype=np.float64) < prob
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

def main():

    B = util.Benchmark()
    (W, H, I) = B.size
    S = randomstate(W, H)

    #np.flush(S) if B.bohrium else None      # Why is this needed?
    B.start()
    R = play(S, I)
    #np.flush(R) if B.bohrium else None      # Why is this needed?
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()

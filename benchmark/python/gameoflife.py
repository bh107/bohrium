from __future__ import print_function
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

def randomstate(height, width, B, prob=0.2):
    state = np.zeros((height+2,width+2))
    state[1:-1,1:-1] = B.random_array((width,height)) < prob
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

    if B.inputfn:
        S = B.load_array()
    else:
        S = randomstate(W, H, B)

    B.start()
    R = play(S, I)
    B.stop()

    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()

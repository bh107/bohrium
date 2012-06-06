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
        # count neighbors
        neighbors = ul + um + ur + ml + mr + ll + lm + lr
        # extract live cells neighbors
        live = neighbors * cells
        # find cells the stay alive
        stay = (live >= SURVIVE_LOW) & (live <= SURVIVE_HIGH) 
        # extract dead cell neighbors
        dead = neighbors * (cells == 0)
        # find cells that spaw new life
        spawn = dead == SPAWN
        # save result for next iteration
        cells[:] = stay | spawn
    return state

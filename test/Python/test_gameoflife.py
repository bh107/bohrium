import numpy as np
import random
import numpytest
import cphvbbridge

dtype = np.int32

def gameoflife(W,H,ITER,DIST,random_state):
    random.setstate(random_state)
    LIVING_LOW= 2
    LIVING_HIGH = 3
    ALIVE = 3

    full = np.zeros((W+2,H+2), dtype=dtype)

    cells = full[1:W+1,1:H+1]
    ul = full[0:W, 0:H]
    um = full[0:W, 1:H+1]
    ur = full[0:W, 2:H+2]
    ml = full[1:W+1, 0:H]
    mr = full[1:W+1, 2:H+2]
    ll = full[2:W+2, 0:H]
    lm = full[2:W+2, 1:H+1]
    lr = full[2:W+2, 2:H+2]

    for i in xrange(W):
      for j in range(H):
          if random.random() > .8:
              cells[i][j] = 1
    if DIST:
        cphvbbridge.handle_array(full)

    for i in xrange(ITER):
        # count neighbors
        neighbors = ul + um + ur + ml + mr + ll + lm + lr
        # extract live cells neighbors
        live = neighbors * cells
        # find all living cells among the already living
        live2 = live == LIVING_LOW
        live = live == LIVING_HIGH
        # merge living cells into 'live'
        live = live | live2
        # extract dead cell neighbors
        dead = cells == 0
        dead *= neighbors
        dead = dead == ALIVE
        # make sure all threads have read their values
        cells = live | dead
    return full

def run():
    random_state = random.getstate()
    Seq = gameoflife(100,100,10,False,random_state)
    Par = gameoflife(100,100,10,True,random_state)
    if not numpytest.array_equal(Seq,Par):
        raise Exception("Incorrect result matrix\n")

if __name__ == "__main__":
    run()

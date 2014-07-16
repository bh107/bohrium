import itertools as it

import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def shape(dims, size=20):
    """
    Generate a suitable N dimensional shape of size
    size**2 core elements
   """
    shape=[]
    for _ in xrange(dims):
        ds = size/dims
        shape.append(2**ds+2)
        dims-=1
        size-=ds

    return shape

def world(shape, core_v, edge_v, dtype=np.float32):
    """
    Generate an N=len(shape) dimensional world with core values
    set to core_v and edge values set to edge_v
    """
    w = np.empty(shape, dtype=dtype)
    w[:] = w.dtype.type(edge_v)
    v = [s for s in it.starmap(slice,it.repeat((1,-1),len(w.shape)))]
    w[v][:] = w.dtype.type(core_v)

    return w

def solve(world, I):
    """
    Run a simple dence stencil operation on a ND array world
    for I iterations
    """
    stencil = [world[s] for s in [map((lambda se : slice(se[0],se[1])),i)
                                  for i in it.product([(0,-2),(1,-1),(2,None)],
                                                      repeat=len(world.shape))]]
    FAC = 1.0/len(stencil)
    for _ in xrange(I):
        stencil[len(stencil)/2][:] = sum(stencil)*FAC
        np.flush()

    return world

if __name__ == "__main__":

    B = util.Benchmark()
    size    = B.size[0]
    I       = B.size[1]
    D       = B.size[2]

    world = np.random.random(shape(D, size), dtype=B.dtype)
    print "Solving",D, "dimensional",world.shape,"problem with", \
        len([i for i in it.product([None,None,None], repeat=D)]), "point stencil."

    B.start()
    solve(world,I)
    B.stop()
    B.pprint()

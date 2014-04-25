import bohrium as np
import itertools as it

def shape(dims,size=20):
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

def world(shape, core_v, edge_v, dtype=np.float32,bohrium=True):
    """
    Generate an N=len(shape) dimensional world with core values
    set to core_v and edge values set to edge_v
    """
    w = np.empty(shape, dtype=dtype, bohrium=bohrium)
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

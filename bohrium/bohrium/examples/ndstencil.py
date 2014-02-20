import bohrium as np
import bohriumbridgemodule as bridge
import itertools as it

def worldND(dims,size=20,dtype=np.float32,bohrium=False):
    """ Generate and ND world of dims dimensions  with 
    size**2 core (random) elements, suitable for solveND """
    shape=[]
    for _ in xrange(dims):
        ds = size/dims
        shape.append(2**ds+2)
        dims-=1
        size-=ds
    return np.random.random(shape,dtype=dtype,bohrium=bohrium)
    
def solveND(world, I):
    """ Run a simple dence stencil operation on a ND array world
    for I iterations  """
    stencil = [world[s] for s in [map((lambda se : slice(se[0],se[1])),i) 
                                  for i in it.product([(0,-2),(1,-1),(2,None)], 
                                                      repeat=len(world.shape))]]
    FAC = 1.0/len(stencil)
    for _ in xrange(I):
        stencil[len(stencil)/2][:] = sum(stencil)*FAC 
        bridge.flush()
    return world

import bohrium as np
import bohriumbridgemodule as bridge
import itertools as it
import util

def worldND(dims,size=20,dtype=np.float32,bohrium=False):
    shape=[]
    for _ in xrange(dims):
        ds = size/dims
        shape.append(2**ds+2)
        dims-=1
        size-=ds
    return np.random.random(shape,dtype=dtype,bohrium=bohrium)
    
def solveND(world, I):
    stencil = [world[s] for s in [map((lambda se : slice(se[0],se[1])),i) 
                                  for i in it.product([(0,-2),(1,-1),(2,None)], 
                                                      repeat=len(world.shape))]]
    FAC = 1.0/len(stencil)
    for _ in xrange(I):
        stencil[len(stencil)/2] = sum(stencil)*FAC 
        bridge.flush()

B = util.Benchmark()
size = B.size[0]
I = B.size[1]
D = B.size[2]

world = worldND(D,size=size,dtype=B.dtype)
world.bohrium=B.bohrium
print "Solving",D, "dimensional",world.shape,"problem with", \
    len([i for i in it.product([None,None,None], repeat=D)]), "point stencil."
B.start()
solveND(world,I)
B.stop()
B.pprint()


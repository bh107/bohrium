import bohrium as np
import bohrium.examples.ndstencil as nds
import itertools as it
import util

B = util.Benchmark()
size = B.size[0]
I = B.size[1]
D = B.size[2]

world = np.random.random(nds.shape(D,size),dtype=B.dtype,bohrium=False)
world.bohrium=B.bohrium
print "Solving",D, "dimensional",world.shape,"problem with", \
    len([i for i in it.product([None,None,None], repeat=D)]), "point stencil."
B.start()
nds.solve(world,I)
B.stop()
B.pprint()


import bohrium as np
import bohriumbridgemodule as bridge
import itertools
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
D = B.size[2]
I = B.size[3]

FAC = 1.0/27

world = np.random.random((H+1)*(W+1)*(D+1),dtype=B.dtype,bohrium=False)
world.shape =(H+1, W+1, D+1)

stencil = [world[1+i[0]:-2+i[0], 1+i[1]:-2+i[1], 1+i[2]:-2+i[2]] \
           for i in itertools.product((-1,0,1), repeat=3)]
world.bohrium=B.bohrium
B.start()
for _ in xrange(I):
        world[1:-2,1:-2,1:-2] = sum(stencil)*FAC
        bridge.flush()
B.stop()
B.pprint()

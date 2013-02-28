import bohrium as np
import bohrium.examples.shallow_water as sw
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

m = sw.model(H,W,dtype=B.dtype,bohrium=B.bohrium)

B.start()
m = sw.simulate(m,I)
B.stop()
B.pprint()

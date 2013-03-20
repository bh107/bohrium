import bohrium as np
import bohrium.examples.sor as sor
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

ft = sor.freezetrap(H,W,dtype=B.dtype,bohrium=B.bohrium)

B.start()
ft = sor.solve(ft,max_iterations=I)
B.stop()
B.pprint()

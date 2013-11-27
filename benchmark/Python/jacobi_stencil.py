import bohrium as np
import bohrium.examples.jacobi_stencil as js
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

ft = js.freezetrap(H,W,dtype=B.dtype,bohrium=B.bohrium)

B.start()
#ft = js.solve(ft,max_iterations=I)
ft = js.iterate(ft, I)
B.stop()
B.pprint()

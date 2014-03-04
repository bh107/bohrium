import bohrium as np
import bohrium.examples.black_scholes as bs
import util

B = util.Benchmark()
N = B.size[0]
I = B.size[1]

S = bs.model(N,dtype=B.dtype,bohrium=B.bohrium)

B.start()
R = bs.price(S,I, visualize=B.visualize)
B.stop()
B.pprint()

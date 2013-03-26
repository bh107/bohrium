import bohrium as np
import bohrium.examples.lattice_boltzmann_D2Q9 as lb
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

cylinder = lb.cylinder(H, W, obstacle=False, bohrium=B.bohrium)

B.start()
lb.solve(cylinder,I)
B.stop()
B.pprint()

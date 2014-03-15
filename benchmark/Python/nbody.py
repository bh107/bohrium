"""NBody in N^2 complexity
Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import bohrium as np
import bohrium.examples.nbody as nb
import util

B = util.Benchmark()
N = B.size[0]
I = B.size[1]

galaxy = nb.random_galaxy( N, B.dtype, B.bohrium)

B.start()
nb.simulate(galaxy,I, visualize=B.visualize)
r = np.add.reduce(galaxy['x'] + galaxy['y'] + galaxy['z'])
B.stop()
B.pprint()


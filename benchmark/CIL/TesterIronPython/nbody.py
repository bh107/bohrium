"""NBody in N^2 complexity
Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import numcil as numpy
import time

# By using the solar-mass as the mass unit and years as the standard time-unit
# the gravitational constant becomes 1

G = 1.0

def fill_diagonal(a, val):
    d,_ = a.shape   #This only makes sense for square matrices
    #a.shape=d*d     #Flatten a without making a copy
    a.reshape((d*d))[::d+1]=val    #Assign the diagonal values
    #a.shape = (d,d) #Return a to its original shape



def calc_force(b):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)/((x_b-x_a)/r)
    """

    dx = b['x'] - b['x'][numpy.newaxis,:].T
    fill_diagonal(dx,1.0)
    dy = b['y'] - b['y'][numpy.newaxis,:].T
    fill_diagonal(dy,1.0)
    dz = b['z'] - b['z'][numpy.newaxis,:].T
    fill_diagonal(dz,1.0)
    pm = b['m'] * b['m'][numpy.newaxis,:].T
    fill_diagonal(pm,0.0)

    r = ( dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    #In the below calc of the the forces the force of a body upon itself
    #becomes nan and thus destroys the data

    Fx = G * pm / r ** 2 * (dx / r)
    Fy = G * pm / r ** 2 * (dy / r)
    Fz = G * pm / r ** 2 * (dz / r)

    #The diagonal nan numbers must be removed so that the force from a body
    #upon itself is zero

    fill_diagonal(Fx,0)
    fill_diagonal(Fy,0)
    fill_diagonal(Fz,0)

    b['vx'] += numpy.add.reduce(Fx, axis=1)/ b['m']
    b['vy'] += numpy.add.reduce(Fy, axis=1)/ b['m']
    b['vz'] += numpy.add.reduce(Fz, axis=1)/ b['m']


def move(galaxy):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

    calc_force(galaxy)

    galaxy['x'] += galaxy['vx']
    galaxy['y'] += galaxy['vy']
    galaxy['z'] += galaxy['vz']


def random_galaxy(
    x_max,
    y_max,
    z_max,
    n,
    bohrium,
    dtype
    ):
    """Generate a galaxy of random bodies"""

    max_mass = 40.0  # Best guess of maximum known star

    # We let all bodies stand still initially

    return {
        'm': numpy.random.random(n,dtype=dtype,bohrium=bohrium) * 10**6 / (4 * numpy.pi ** 2),
        'x': numpy.random.random(n,dtype=dtype,bohrium=bohrium)*2*x_max-x_max,
        'y': numpy.random.random(n,dtype=dtype,bohrium=bohrium)*2*x_max-x_max,
        'z': numpy.random.random(n,dtype=dtype,bohrium=bohrium)*2*x_max-x_max,
        'vx': numpy.empty(n,dtype=dtype,bohrium=bohrium) * 0.0,
        'vy': numpy.empty(n,dtype=dtype,bohrium=bohrium) * 0.0,
        'vz': numpy.empty(n,dtype=dtype,bohrium=bohrium) * 0.0,
        }

def Solve_nBody(size = None):
    import util
    B = util.Benchmark(size)
    bodies = B.size[0]
    time_step = B.size[1]

    x_max = 500
    y_max = 500
    z_max = 500

    galaxy = random_galaxy(x_max, y_max, z_max, bodies, B.bohrium, B.dtype)

    B.start()
    for _ in range(time_step):
        move(galaxy)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    Solve_nBody()




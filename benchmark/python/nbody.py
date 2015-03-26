from __future__ import print_function
"""
NBody in N^2 complexity

Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

G     = 6.67384e-11     # m/(kg*s^2)
dt    = 60*60*24*365.25 # Years in seconds
r_ly  = 9.4607e15       # Lightyear in m
m_sol = 1.9891e30       # Solar mass in kg

def diagonal(ary, offset=0):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.

    Parameters
    ----------
    ary : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, a 1-D array containing the diagonal is returned.
        If the dimension of `a` is larger, then an array of diagonals is
        returned, "packed" from left-most dimension to right-most (e.g.,
        if `a` is 3-D, then the diagonals are "packed" along rows).

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : MATLAB work-a-like for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    """
    if ary.ndim != 2:
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (ary.shape[0]-offset) > ary.shape[1]:
            ary_diag = ary[offset, :]
        else:
            ary_diag = ary[offset:, 0]
    else:
        if ary.shape[1]-offset > ary.shape[0]:
            ary_diag = ary[:, offset]
        else:
            ary_diag = ary[0, offset:]
    ary_diag.strides = (ary.strides[0]+ary.strides[1],)
    return ary_diag


def random_galaxy(N, B, dtype=np.float64):
    """Generate a galaxy of random bodies"""

    galaxy = {            # We let all bodies stand still initially
        'm':    (B.random_array((N,), dtype=dtype) + dtype(10)) * dtype(m_sol/10),
        'x':    (B.random_array((N,), dtype=dtype) - dtype(0.5)) * dtype(r_ly/100),
        'y':    (B.random_array((N,), dtype=dtype) - dtype(0.5)) * dtype(r_ly/100),
        'z':    (B.random_array((N,), dtype=dtype) - dtype(0.5)) * dtype(r_ly/100),
        'vx':   np.zeros(N, dtype=dtype),
        'vy':   np.zeros(N, dtype=dtype),
        'vz':   np.zeros(N, dtype=dtype)
    }
    if dtype == np.float32:
        galaxy['m'] /= 1e10
        galaxy['x'] /= 1e5
        galaxy['y'] /= 1e5
        galaxy['z'] /= 1e5
    return galaxy

def move(galaxy, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """
    n  = len(galaxy['x'])
    # Calculate all dictances component wise (with sign)
    dx = galaxy['x'][np.newaxis,:].T - galaxy['x']
    dy = galaxy['y'][np.newaxis,:].T - galaxy['y']
    dz = galaxy['z'][np.newaxis,:].T - galaxy['z']

    # Euclidian distances (all bodys)
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    diagonal(r)[:] = 1.0

    # prevent collition
    mask = r < 1.0
    r = r * ~mask + 1.0 * mask

    m = galaxy['m'][np.newaxis,:].T

    # Calculate the acceleration component wise
    Fx = G*m*dx/r**3
    Fy = G*m*dy/r**3
    Fz = G*m*dz/r**3
    # Set the force (acceleration) a body exerts on it self to zero
    diagonal(Fx)[:] = 0.0
    diagonal(Fy)[:] = 0.0
    diagonal(Fz)[:] = 0.0

    galaxy['vx'] += dt*np.sum(Fx, axis=0)
    galaxy['vy'] += dt*np.sum(Fy, axis=0)
    galaxy['vz'] += dt*np.sum(Fz, axis=0)

    galaxy['x'] += dt*galaxy['vx']
    galaxy['y'] += dt*galaxy['vy']
    galaxy['z'] += dt*galaxy['vz']

def simulate(galaxy, timesteps, visualize=False):
    for i in xrange(timesteps):
        move(galaxy,dt)
        if visualize:#NB: this is only for experiments
            T = np.zeros((3, len(galaxy['x'])), dtype=np.float32)
            T[0,:] = galaxy['x']
            T[1,:] = galaxy['y']
            T[2,:] = galaxy['z']
            np.visualize(T, "3d", 0, 0.0, 10)

def main():
    B = util.Benchmark()
    N = B.size[0]
    I = B.size[1]

    if B.inputfn:
        galaxy = B.load_arrays(B.inputfn)
    else:
        galaxy = random_galaxy(N, B, B.dtype)

    if B.dumpinput:
        B.dump_arrays("nbody", galaxy)

    B.start()
    simulate(galaxy, I, visualize=B.visualize)
    R = galaxy['x'] + galaxy['y'] + galaxy['z']
    B.stop()

    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res':R})

if __name__ == "__main__":
    main()

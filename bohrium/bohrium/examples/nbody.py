import bohrium as np

G     = 6.67384e-11     # m/(kg*s²)
dt    = 60*60*24*365.25 # Years in seconds
r_ly  = 9.4607e15       # Lightyear in m
m_sol = 1.9891e30       # Solar mass in kg

def random_galaxy( N, dtype=np.float64, bohrium=True):
    """Generate a galaxy of random bodies"""
    galaxy = {            # We let all bodies stand still initially
        'm':    (np.random.random(N, dtype=dtype, bohrium=bohrium) + dtype(10)) * dtype(m_sol/10),
        'x':    (np.random.random(N, dtype=dtype, bohrium=bohrium) - dtype(0.5)) * dtype(r_ly/100),
        'y':    (np.random.random(N, dtype=dtype, bohrium=bohrium) - dtype(0.5)) * dtype(r_ly/100),
        'z':    (np.random.random(N, dtype=dtype, bohrium=bohrium) - dtype(0.5)) * dtype(r_ly/100),
        'vx':   np.zeros(N, dtype=dtype, bohrium=bohrium),
        'vy':   np.zeros(N, dtype=dtype, bohrium=bohrium),
        'vz':   np.zeros(N, dtype=dtype, bohrium=bohrium)
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
    np.diagonal(r)[:] = 1.0

    # prevent collition
    mask = r < 1.0
    r = r * ~mask + 1.0 * mask

    m = galaxy['m'][np.newaxis,:].T

    # Calculate the acceleration component wise
    Fx = G*m*dx/r**3
    Fy = G*m*dy/r**3
    Fz = G*m*dz/r**3
    # Set the force (acceleration) a body exerts on it self to zero
    np.diagonal(Fx)[:] = 0.0
    np.diagonal(Fy)[:] = 0.0
    np.diagonal(Fz)[:] = 0.0

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


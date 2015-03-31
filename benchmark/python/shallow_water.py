from __future__ import print_function
"""
Shallow Water
-------------

So what does this code example illustrate?

Adapted from: http://people.sc.fsu.edu/~jburkardt/m_src/shallow_water_2d/
"""
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

g = 9.80665 # gravitational acceleration

def droplet(height, width, data_type=np.float32):
    """Generate grid of droplets"""

    x = np.array(np.linspace(-1, 1, num=width, endpoint=True), dtype=data_type)
    y = np.array(np.linspace(-1, 1, num=width, endpoint=True), dtype=data_type)

    (xx, yy) = np.meshgrid(x, y)

    droplet = height * np.exp(-5 * (xx ** 2 + yy ** 2))

    return droplet

def model(height, width, dtype=np.float32):
    assert height >= 16
    assert width >= 16

    m = np.ones((height, width), dtype=dtype)
    D = droplet(8, 8)  # simulate a water drop
    droploc = height / 4
    (dropx, dropy) = D.shape
    m[droploc:droploc + dropx, droploc:droploc + dropy] += D
    droploc = height / 2
    (dropx, dropy) = D.shape
    m[droploc:droploc + dropx, droploc:droploc + dropy] += D

    return m

#FLOP count: i*(12*s + 4*s**2 + 14*s**2 + 9*s**2 + 4*s**2 + 9*s**2 + 14*s**2 + 6*s**2 + 19*s**2 + 19*s**2)
#where s is size and i is iterations
def step(H, U, V, dt=0.02, dx=1.0, dy=1.0):
    # Reflecting boundary conditions
    H[:,0] = H[:,1]   ; U[:,0] = U[:,1]     ; V[:,0] = -V[:,1]
    H[:,-1] = H[:,-2] ; U[:,-1] = U[:,-2]   ; V[:,-1] = -V[:,-2]
    H[0,:] = H[1,:]   ; U[0,:] = -U[1,:]    ; V[0,:] = V[1,:]
    H[-1,:] = H[-2,:] ; U[-1,:] = -U[-2,:]  ; V[-1,:] = V[-2,:]

    #First half step

    # height
    Hx = (H[1:,1:-1]+H[:-1,1:-1])/2 - dt/(2*dx)*(U[1:,1:-1]-U[:-1,1:-1])

    # x momentum
    Ux = (U[1:,1:-1]+U[:-1,1:-1])/2 - \
         dt/(2*dx) * ((U[1:,1:-1]**2/H[1:,1:-1] + g/2*H[1:,1:-1]**2) -
                      (U[:-1,1:-1]**2/H[:-1,1:-1] + g/2*H[:-1,1:-1]**2))

    # y momentum
    Vx = (V[1:,1:-1]+V[:-1,1:-1])/2 - \
         dt/(2*dx) * ((U[1:,1:-1]*V[1:,1:-1]/H[1:,1:-1]) -
                      (U[:-1,1:-1]*V[:-1,1:-1]/H[:-1,1:-1]))

    # height
    Hy = (H[1:-1,1:]+H[1:-1,:-1])/2 - dt/(2*dy)*(V[1:-1,1:]-V[1:-1,:-1])

    #x momentum
    Uy = (U[1:-1,1:]+U[1:-1,:-1])/2 - \
         dt/(2*dy)*((V[1:-1,1:]*U[1:-1,1:]/H[1:-1,1:]) -
                    (V[1:-1,:-1]*U[1:-1,:-1]/H[1:-1,:-1]))
    #y momentum
    Vy = (V[1:-1,1:]+V[1:-1,:-1])/2 - \
         dt/(2*dy)*((V[1:-1,1:]**2/H[1:-1,1:] + g/2*H[1:-1,1:]**2) -
                    (V[1:-1,:-1]**2/H[1:-1,:-1] + g/2*H[1:-1,:-1]**2))

    #Second half step

    # height
    H[1:-1,1:-1] -= (dt/dx)*(Ux[1:,:]-Ux[:-1,:]) + (dt/dy)*(Vy[:,1:]-Vy[:,:-1])

    # x momentum
    U[1:-1,1:-1] -= (dt/dx)*((Ux[1:,:]**2/Hx[1:,:] + g/2*Hx[1:,:]**2) -
                             (Ux[:-1,:]**2/Hx[:-1,:] + g/2*Hx[:-1,:]**2)) + \
                    (dt/dy)*((Vy[:,1:] * Uy[:,1:]/Hy[:,1:]) -
                             (Vy[:,:-1] * Uy[:,:-1]/Hy[:,:-1]))
    # y momentum
    V[1:-1,1:-1] -= (dt/dx)*((Ux[1:,:] * Vx[1:,:]/Hx[1:,:]) -
                             (Ux[:-1,:]*Vx[:-1,:]/Hx[:-1,:])) + \
                    (dt/dy)*((Vy[:,1:]**2/Hy[:,1:] + g/2*Hy[:,1:]**2) -
                             (Vy[:,:-1]**2/Hy[:,:-1] + g/2*Hy[:,:-1]**2))
    return (H, U, V)

def simulate(H, timesteps, visualize=False):
    U = np.zeros_like(H)
    V = np.zeros_like(H)
    for i in xrange(timesteps):
        (H, U, V) = step(H, U, V)
        if visualize:
            np.visualize(H, "3d", 0, 0.0, 5.5)
    return H

def main():
    B = util.Benchmark()
    H = B.size[0]
    W = B.size[1]
    I = B.size[2]

    if B.inputfn:
        M = B.load_array()
    else:
        M = model(H, W, dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("shallow_water", {'input':M})

    B.start()
    M = simulate(M, I, visualize=B.visualize)
    B.stop()
    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': M})

if __name__ == "__main__":
    main()

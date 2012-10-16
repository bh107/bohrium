# Adapted from: http://people.sc.fsu.edu/~jburkardt/m_src/shallow_water_2d/
# Saved images may be converted into an animated gif with:
# convert   -delay 20   -loop 0   swater*.png   swater.gif

import cphvbnumpy as numpy
import time
import util

B = util.Benchmark()
n = B.size[0]
T = B.size[1]

g   = 9.8       # gravitational constant
dt  = 0.02      # hardwired timestep
dx  = 1.0
dy  = 1.0
droploc = n/4

H   = numpy.ones((n+2,n+2),     dtype=B.dtype);
U   = numpy.zeros((n+2,n+2),    dtype=B.dtype);
V   = numpy.zeros((n+2,n+2),    dtype=B.dtype);
Hx  = numpy.zeros((n+1,n+1),    dtype=B.dtype);
Ux  = numpy.zeros((n+1,n+1),    dtype=B.dtype);
Vx  = numpy.zeros((n+1,n+1),    dtype=B.dtype);
Hy  = numpy.zeros((n+1,n+1),    dtype=B.dtype);
Uy  = numpy.zeros((n+1,n+1),    dtype=B.dtype);
Vy  = numpy.zeros((n+1,n+1),    dtype=B.dtype);

H[droploc,droploc] += 5.0

H.cphvb     = B.cphvb
Hx.cphvb    = B.cphvb
Hy.cphvb    = B.cphvb
U.cphvb     = B.cphvb
Ux.cphvb    = B.cphvb
Uy.cphvb    = B.cphvb
V.cphvb     = B.cphvb
Vx.cphvb    = B.cphvb
Vy.cphvb    = B.cphvb

B.start()
for i in xrange(T):

    # Reflecting boundary conditions
    #               - equiv
    H[:,0] = H[:,1]   ; U[:,0] = U[:,1]     ; V[:,0] = -V[:,1]
    H[:,n+1] = H[:,n] ; U[:,n+1] = U[:,n]   ; V[:,n+1] = -V[:,n]
    H[0,:] = H[1,:]   ; U[0,:] = -U[1,:]    ; V[0,:] = V[1,:]
    H[n+1,:] = H[n,:] ; U[n+1,:] = -U[n,:]  ; V[n+1,:] = V[n,:]

    #
    # First half step
    #
    # height        - score
    Hx[:,:-1] = (H[1:,1:-1]+H[:-1,1:-1])/2 - \
                dt/(2*dx)*(U[1:,1:-1]-U[:-1,1:-1]);

    # x momentum    - simple
    Ux[:,:-1] = (U[1:,1:-1]+U[:-1,1:-1])/2 - \
                dt/(2*dx)*((U[1:,1:-1]**2/H[1:,1:-1] + \
                            g/2*H[1:,1:-1]**2) - \
                           (U[:-1,1:-1]**2/H[:-1,1:-1] + \
                            g/2*H[:-1,1:-1]**2))
    
    # y momentum    - equiv
    Vx[:,:-1] = (V[1:,1:-1]+V[:-1,1:-1])/2 - \
                dt/(2*dx)*((U[1:,1:-1] * \
                            V[1:,1:-1]/H[1:,1:-1]) - \
                           (U[:-1,1:-1] * \
                            V[:-1,1:-1]/H[:-1,1:-1]))
    
    #height         - score
    Hy[:-1,:] = (H[1:-1,1:]+H[1:-1,:-1])/2 - \
                dt/(2*dy)*(V[1:-1,1:]-V[1:-1,:-1])

    #x momentum     - equiv
    Uy[:-1,:] = (U[1:-1,1:]+U[1:-1,:-1])/2 - \
                dt/(2*dy)*((V[1:-1,1:] * \
                            U[1:-1,1:]/H[1:-1,1:]) - \
                           (V[1:-1,:-1] * \
                            U[1:-1,:-1]/H[1:-1,:-1]))

    #y momentum     - simple
    Vy[:-1,:] = (V[1:-1,1:]+V[1:-1,:-1])/2 - \
                dt/(2*dy)*((V[1:-1,1:]**2/H[1:-1,1:] + \
                            g/2*H[1:-1,1:]**2) - \
                           (V[1:-1,:-1]**2/H[1:-1,:-1] + \
                            g/2*H[1:-1,:-1]**2))

    # Second half step

    # height        - score
    H[1:-1,1:-1] = H[1:-1,1:-1] - \
                   (dt/dx)*(Ux[1:,:-1]-Ux[:-1,:-1]) - \
                   (dt/dy)*(Vy[:-1,1:]-Vy[:-1,:-1])

    # x momentum    - score
    U[1:-1,1:-1] = U[1:-1,1:-1] - \
                   (dt/dx)*((Ux[1:,:-1]**2/Hx[1:,:-1] + \
                             g/2*Hx[1:,:-1]**2) - \
                            (Ux[:-1,:-1]**2/Hx[:-1,:-1] + \
                             g/2*Hx[:-1,:-1]**2)) - \
                             (dt/dy)*((Vy[:-1,1:] * \
                                       Uy[:-1,1:]/Hy[:-1,1:]) - \
                                        (Vy[:-1,:-1] * \
                                         Uy[:-1,:-1]/Hy[:-1,:-1]))
    
    # y momentum    - score
    V[1:-1,1:-1] = V[1:-1,1:-1] - \
                   (dt/dx)*((Ux[1:,:-1] * \
                             Vx[1:,:-1]/Hx[1:,:-1]) - \
                            (Ux[:-1,:-1]*Vx[:-1,:-1]/Hx[:-1,:-1])) - \
                            (dt/dy)*((Vy[:-1,1:]**2/Hy[:-1,1:] + \
                                      g/2*Hy[:-1,1:]**2) - \
                                     (Vy[:-1,:-1]**2/Hy[:-1,:-1] + \
                                      g/2*Hy[:-1,:-1]**2))

res = numpy.add.reduce(numpy.add.reduce(H / n))

B.stop()
B.pprint()


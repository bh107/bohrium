#!/usr/bin/python
# -*- coding: utf-8 -*-

"""NBody in N^2 complexity
Note that we are unp.sing only Newtonian forces and do not consider relativity
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

G = 6.673e-11
solarmass=1.98892e30


def fill_diagonal(a, val):
    d,_ = a.shape   #This only makes sense for square matrices
    a.shape=d*d     #Flatten a without making a copy
    a[::d+1]=val    #Assign the diagonal values
    a.shape = (d,d) #Return a to its original shape

def calc_force(a, b, dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)/((x_b-x_a)/r)
    """

    dx = b['x'] - a['x'][np.newaxis,:].T
    dy = b['y'] - a['y'][np.newaxis,:].T
    dz = b['z'] - a['z'][np.newaxis,:].T
    pm = b['m'] * a['m'][np.newaxis,:].T

    if a is b:
        fill_diagonal(dx,1.0)
        fill_diagonal(dy,1.0)
        fill_diagonal(dz,1.0)
        fill_diagonal(pm,0.0)

    r = ( dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    #In the below calc of the the forces the force of a body upon itself
    #becomes nan and thus destroys the data

    Fx = G * pm / r ** 2 * (dx / r)
    Fy = G * pm / r ** 2 * (dy / r)
    Fz = G * pm / r ** 2 * (dz / r)

    #The diagonal nan numbers must be removed so that the force from a body
    #upon itself is zero
    if a is b:
        fill_diagonal(Fx,0)
        fill_diagonal(Fy,0)
        fill_diagonal(Fz,0)

    a['vx'] += np.add.reduce(Fx, axis=1)/ a['m'] * dt
    a['vy'] += np.add.reduce(Fy, axis=1)/ a['m'] * dt
    a['vz'] += np.add.reduce(Fz, axis=1)/ a['m'] * dt

def move(solarsystem, astoroids, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

    calc_force(solarsystem, solarsystem, dt)
    calc_force(astoroids, solarsystem, dt)

    solarsystem['x'] += solarsystem['vx'] * dt
    solarsystem['y'] += solarsystem['vy'] * dt
    solarsystem['z'] += solarsystem['vz'] * dt

    astoroids['x'] += astoroids['vx'] * dt
    astoroids['y'] += astoroids['vy'] * dt
    astoroids['z'] += astoroids['vz'] * dt

def circlev(rx, ry, rz):
    r2=np.sqrt(rx*rx+ry*ry+rz*rz)
    numerator=(6.67e-11)*1e6*solarmass
    return np.sqrt(numerator/r2)

def sign(x):
    if x<0: return -1
    if x>0: return 1
    return 0

def random_system(x_max, y_max, z_max, n, b, B):
    """Generate a galaxy of random bodies"""

    solarsystem = {'m':np.empty(n), 'x':np.empty(n), 'y':np.empty(n),'z':np.empty(n),\
                   'vx':np.empty(n), 'vy':np.empty(n),'vz':np.empty(n)}
    solarsystem['m'][0]= 1e6*solarmass
    solarsystem['x'][0]= 0
    solarsystem['y'][0]= 0
    solarsystem['z'][0]= 0
    solarsystem['vx'][0]= 0
    solarsystem['vy'][0]= 0
    solarsystem['vz'][0]= 0
    for i in xrange(1,n):
        px, py,pz = B.random_array((1,)), B.random_array((1,)), B.random_array((1,))*.01
        dist = (1.0/np.sqrt(px*px+py*py+pz*pz))-(.8-B.random_array((1,))*.1)
        px = x_max*px*dist*sign(.5-B.random_array((1,)))
        py = y_max*py*dist*sign(.5-B.random_array((1,)))
        pz = z_max*pz*dist*sign(.5-B.random_array((1,)))
        solarsystem['x'][i], solarsystem['y'][i], solarsystem['z'][i] = px, py, pz
        magv = circlev(px,py, pz)

        absangle = np.arctan(abs(py/px))
        thetav= np.pi/2-absangle
        vx   = -1*sign(py)*np.cos(thetav)*magv
        vy   = sign(px)*np.sin(thetav)*magv
        vz   = 0
        solarsystem['vx'][i], solarsystem['vy'][i], solarsystem['vz'][i] = vx, vy, vz
        solarsystem['m'][i] = B.random_array((1,))*solarmass*10+1e20;

    astoroids = {'m':np.empty(b), 'x':np.empty(b), 'y':np.empty(b),'z':np.empty(b),\
                 'vx':np.empty(b), 'vy':np.empty(b),'vz':np.empty(b)}
    for i in xrange(b):
        px, py,pz = B.random_array((1,)), B.random_array((1,)), B.random_array((1,))*.01
        dist = (1.0/np.sqrt(px*px+py*py+pz*pz))-(B.random_array((1,))*.1)
        px = x_max*px*dist*sign(.5-B.random_array((1,)))
        py = y_max*py*dist*sign(.5-B.random_array((1,)))
        pz = z_max*pz*dist*sign(.5-B.random_array((1,)))
        astoroids['x'][i], astoroids['y'][i], astoroids['z'][i] = px, py, pz
        magv = circlev(px,py, pz)

        absangle = np.arctan(abs(py/px))
        thetav= np.pi/2-absangle
        vx   = -1*sign(py)*np.cos(thetav)*magv
        vy   = sign(px)*np.sin(thetav)*magv
        vz   = 0
        astoroids['vx'][i], astoroids['vy'][i], astoroids['vz'][i] = vx, vy, vz

        astoroids['m'][i] = B.random_array((1,))*solarmass*10+1e14;

    return solarsystem, astoroids


def gfx_init(xm, ym, zm):
    """Init plot"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

    plt.ion()
    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.xm = xm
    sub.ym = ym
    sub.zm = zm
    return sub


def show(sub, solarsystem, bodies):
    """Show plot"""
    import matplotlib.pyplot as plt
    #Sun
    sub.clear()

    sub.scatter(
                solarsystem['x'][0],
                solarsystem['y'][0],
                solarsystem['z'][0],
                s=100,
                marker='o',
                c='yellow',
            )
    #Planets
    sub.scatter(
                [solarsystem['x'][1:]],
                [solarsystem['y'][1:]],
                [solarsystem['z'][1:]],
                s=5,
                marker='o',
                c='blue',
        )


#Astoroids
    sub.scatter(
                [bodies['x']],
                [bodies['y']],
                [bodies['z']],
                s=.1,
                marker='.',
                c='green',
        )


    sub.set_xbound(-sub.xm, sub.xm)
    sub.set_ybound(-sub.ym, sub.ym)
    try:
        sub.set_zbound(-sub.zm, sub.zm)
    except AttributeError:
        print 'Warning: correct 3D plots may require matplotlib-1.1 or later'

    plt.draw()

def main():
    B = util.Benchmark()
    num_asteroids  = B.size[0]
    num_planets    = B.size[1]
    num_iteratinos = B.size[2]

    x_max = 1e18
    y_max = 1e18
    z_max = 1e18
    dt = 1e12

    print("INITIALIZING SYSTEM")
    solarsystem, astoroids = random_system(x_max, y_max, z_max, num_planets, num_asteroids, B)
    if B.verbose:
        P3 = gfx_init(x_max, y_max, z_max)
    print("I WILL START NOW")
    B.start()
    for _ in range(num_iteratinos):
        move(solarsystem, astoroids, dt)
        if B.verbose:
            show(P3, solarsystem, astoroids)
    R = solarsystem['x']
    B.stop()
    B.pprint()
    if B.verbose:
        print(R)
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()

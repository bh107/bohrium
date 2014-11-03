from __future__ import print_function
from time import time
from sys import argv
import bohrium as np

def window(B,a=0.37):
    wl = np.ones_like(B[0])
    b = int(np.ceil((a * (n-1) / 2)))
    wl[:b] =  0.5 * (1 + np.cos(np.pi*(2 * np.arange(b) / (a * (n-1)) - 1)))
    wl[-b:] =  0.5 * (1 + np.cos(np.pi*(2 * np.arange(b-1,-1,-1) / (a * (n-1)) - 1)))
    w = np.sqrt(wl**2+wl[:,None]**2)
    return B*w

def calcB(B_x0, alpha=0.0,
          x_min = 0.0, x_max = 0.25,
          y_min = 0.0, y_max = 1.0,
          z_min = 0.0, z_max = 1.0):

    n = len(B_x0)
    start = time()
    x = np.linspace(x_min,x_max,num=n,endpoint=False,dtype=B_x0.dtype)                           # n
    y = np.linspace(y_min,y_max,num=n,dtype=B_x0.dtype)                                          # n
    z = np.linspace(z_min,z_max,num=n,dtype=B_x0.dtype)                                          # n
    u = np.arange(n,dtype=B_x0.dtype)                                                            # n

    # Making C
    C = np.empty_like(B_x0)                                                                      # n²
    sinuy = np.sin(np.pi/y_max * u[:,None] * y)                                                  #-n²
    sinuz = np.sin(np.pi/z_max * u[:,None] * z)                                                  #-n²
    for i in xrange(n):
        C[i,:] =  4.0 / (n-1.0)**2 * np.sum(np.sum(B_x0 * sinuy[i][:,None] * sinuz[:,None],2),1) #-n³
    del sinuy
    del sinuz
    l = np.pi**2 * ((u**2 / y_max)[:,None] + (u**2 / z_max))                                     # n²
    l[0,0] = 1.0
    r = np.sqrt(l - alpha**2)                                                                    # n²

    # Calculating B
    Bx = np.empty((n,n,n),dtype=B_x0.dtype,bohrium=False)
    By = np.empty((n,n,n),dtype=B_x0.dtype,bohrium=False)
    Bz = np.empty((n,n,n),dtype=B_x0.dtype,bohrium=False)

    exr = np.exp(-r * x[:,None,None])                                                            # n³
    sinuz = np.sin(np.pi/z_max * u * z[:,None])                                                  # n²
    cosuz = np.cos(np.pi/z_max * u * z[:,None])                                                  # n²
    sinuy = np.sin(np.pi/y_max * u * y[:,None])                                                  # n²
    cosuy = np.cos(np.pi/y_max * u * y[:,None])                                                  # n²
    np.flush()
    print("Setup calc:", time() -start)
    for i in xrange(n):
        start = time()
        for j in xrange(n):
            sincos = sinuy[i][:,None] * (u * cosuz[j])                                          # n²
            cossin = (u * cosuy[i])[:,None] * sinuz[j]                                          # n²
            temp_x = C * sinuy[i][:,None] * sinuz[j]                                            # n²
            Cl = C/l                                                                            # n²
            temp_y = Cl * (alpha * np.pi / z_max * sincos - np.pi / y_max * r * cossin)         # n²
            temp_z = Cl * (alpha * np.pi / y_max * cossin + np.pi / z_max * r * sincos)         # n²
            del Cl
            del cossin
            del sincos
            bxij = np.sum(np.sum(temp_x * exr,2),1)                                             # n³
            del temp_x
            bxij.bohrium=False
            Bx[:,i,j] = bxij
            byij = np.sum(np.sum(temp_y * exr,2),1)                                             # n³
            del temp_y
            byij.bohrium=False
            By[:,i,j] = byij
            bzij = np.sum(np.sum(temp_z * exr,2),1)                                             # n³
            del temp_z
            bzij.bohrium=False
            Bz[:,i,j] = bzij
        print("Outer loop:", time()-start)
    return (Bx, By, Bz)

if __name__ == '__main__':
    fname = 'large.dat'
    dtype = np.float64
    #B_x0 = np.loadtxt(fname,dtype=dtype)
    n = 64
    B_x0 = (np.random.random((n,n),bohrium=False) -0.5) * 20.0
    B_x0.bohrium = True
    start = time()
    B_x0 = window(B_x0)
    (Bx, By, Bz) = calcB(B_x0)

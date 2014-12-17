from __future__ import print_function
from time import time
import numpy as np

def window(B,a=0.37):
    w = np.zeros_like(B)
    wl = np.zeros_like(B[0])
    for i in range(n):
        if (i >= 0) and (i < (a * (n-1) / 2)):
            wl[i] = 0.5 * (1 + np.cos(np.pi*(2 * i / (a * (n-1)) - 1)))
        if (i >= (a * (n-1) / 2)) and (i <= (n-1) * (1-a/2)):
            wl[i] = 1
        if (i >= (n-1) * (1-a/2)) and (i <= n-1):
            wl[i] = 0.5 * (1 + np.cos(2 * np.pi * i / (a * (n-1)) - 2 * np.pi / a + np.pi))
    for i in range(n):
        for j in range(n):
            w[i,j] = np.sqrt(wl[i]**2 + wl[j]**2)
    return B * w

def calcB(B_x0, alpha=0.0,
          x_min = 0.0, x_max = 0.25,
          y_min = 0.0, y_max = 1.0,
          z_min = 0.0, z_max = 1.0):

    n = len(B_x0)

    x = np.linspace(x_min,x_max,num=n,endpoint=False).astype(B_x0.dtype,copy=False)
    y = np.linspace(y_min,y_max,num=n).astype(B_x0.dtype,copy=False)
    z = np.linspace(z_min,z_max,num=n).astype(B_x0.dtype,copy=False)
    u = np.arange(n,dtype=B_x0.dtype)

    #Making C
    C = np.empty_like(B_x0)
    l = np.empty_like(B_x0)

    for i in range(n):
        l[i,:] =  np.pi**2 * ((u[i]**2 / y_max) + (u**2 / z_max))
        for j in range(n):
            C[i,j] =  4.0 / (n-1)**2 * np.sum(B_x0 *
                                              np.sin(u[i] * np.pi * y[:,None] / y_max) *
                                              np.sin(u[j] * np.pi * z / z_max))
    l[0,0] = 1.0

    r = np.sqrt(l - alpha**2)


    # Calculating B
    Bx = np.empty((n,n,n),dtype=B_x0.dtype)
    By = np.empty((n,n,n),dtype=B_x0.dtype)
    Bz = np.empty((n,n,n),dtype=B_x0.dtype)

    for i in range(n):
        print (i)
        for j in range(n):
            sincos = np.sin(np.pi * u[:,None] * y[i] / y_max) * (u * np.cos(np.pi * u * z[j] / z_max))
            cossin = (u[:,None] * np.cos(np.pi * u[:,None] * y[i] / y_max)) * (np.sin(np.pi * u * z[j] / z_max))
            temp_x = C * (np.sin(np.pi * u[:,None] * y[i] / y_max) * (np.sin(np.pi * u * z[j] / z_max)))
            temp_y = C / l * (alpha * np.pi / z_max * sincos - r * np.pi / y_max * cossin)
            temp_z = C / l * (alpha * np.pi / y_max * cossin + r * np.pi / z_max * sincos)

            Bx[:,i,j] = np.sum(temp_x * np.exp(-r * x[:,None,None]),(1,2))
            By[:,i,j] = np.sum(temp_y * np.exp(-r * x[:,None,None]),(1,2))
            Bz[:,i,j] = np.sum(temp_z * np.exp(-r * x[:,None,None]),(1,2))
    return (Bx, By, Bz)

if __name__ == '__main__':
    fname = 'large.dat'
    dtype = np.float64
    #B_x0 = np.loadtxt(fname,dtype=dtype)
    n = 64
    B_x0 = (np.random.random((n,n)) -0.5) * 20.0
    B_x0 = window(B_x0)
    (Bx, By, Bz) = calcB(B_x0)

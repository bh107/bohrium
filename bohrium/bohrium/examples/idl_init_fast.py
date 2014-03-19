#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np

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
    
    x = np.linspace(x_min,x_max,num=n,endpoint=False).astype(B_x0.dtype,copy=False)
    y = np.linspace(y_min,y_max,num=n).astype(B_x0.dtype,copy=False)
    z = np.linspace(z_min,z_max,num=n).astype(B_x0.dtype,copy=False)
    u = np.arange(n,dtype=B_x0.dtype)

    # Making C
    C = np.empty_like(B_x0)
    sinuj = np.sin(u[:,None] * np.pi * z / z_max)
    for i in range(n):
        sinui = np.sin(u[i] * np.pi * y[:,None] / y_max)
        for j in range(n):
            C0[i,j] =  4.0 / (n-1)**2 * np.sum(B_x0 * sinui * sinuj[j])
    l = np.pi**2 * ((u[:,None]**2 / y_max) + (u**2 / z_max))
    l[0,0] = 1.0
    r = np.sqrt(l - alpha**2)

    # Calculating B
    Bx = np.empty((n,n,n),dtype=B_x0.dtype)
    By = np.empty((n,n,n),dtype=B_x0.dtype)
    Bz = np.empty((n,n,n),dtype=B_x0.dtype)
    
    exr = np.exp(-r * x[:,None,None])
    sinz = np.sin(np.pi/z_max * u * z[:,None])
    cosz = np.cos(np.pi/z_max * u * z[:,None])
    for i in range(n):
        sinyi = np.sin(np.pi * y[i] / y_max * u)
        cosyi = np.cos(np.pi * y[i] / y_max * u)
        for j in range(n):
            sincos = sinyi[:,None] * (u * cosz[j])
            cossin = (u * cosyi)[:,None] * sinz[j]
            temp_x = C * sinyi[:,None] * sinz[j]
            Cl = C/l
            temp_y = Cl * (alpha * np.pi / z_max * sincos - np.pi / y_max * r * cossin)
            temp_z = Cl * (alpha * np.pi / y_max * cossin + np.pi / z_max * r * sincos)
            Bx[:,i,j] = np.sum(np.sum(temp_x * exr,2),1)
            By[:,i,j] = np.sum(np.sum(temp_y * exr,2),1)
            Bz[:,i,j] = np.sum(np.sum(temp_z * exr,2),1)
    return (Bx, By, Bz)

if __name__ == '__main__':
    fname = 'large.dat'
    dtype = np.float64
    #B_x0 = np.loadtxt(fname,dtype=dtype)
    n = 64
    B_x0 = (np.random.random((n,n)) -0.5) * 20.0
    B_x0 = window(B_x0)
    (Bx, By, Bz) = calcB(B_x0)

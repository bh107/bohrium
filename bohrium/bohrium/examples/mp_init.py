#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from time import time
import numpy as np

def main(fname):
    print 'Loading data'
    B_x0 = np.loadtxt(fname)
    
    alpha = 0
    n = len(B_x0)
    
    print 'Allocating RAM space'
    x = np.arange(n) / (4 * n)
    
    y_min = 0.0
    y_max = 1.0
    y = np.arange(n) / (n-1) * (y_max - y_min) + y_min
    
    z_min = 0.0
    z_max = 1.0
    z = np.arange(n) / (n-1) * (z_max - z_min) + z_min
    
    u = np.arange(n)
    
    Bx = np.zeros((n,n,n))
    By = np.zeros((n,n,n))
    Bz = np.zeros((n,n,n))
    
    print 'Building windowing function'
    wl = np.zeros(n)
    w = np.zeros((n,n))
    a = 0.37
    
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
    
    B_x0 = B_x0 * w
    
    
    #Making C
    print 'Calculating C'
    C = np.zeros((n,n))
    l = np.zeros((n,n))
    
    for i in range(n):
        l[i,:] =  np.pi**2 * ((u[i]**2 / y_max) + (u**2 / z_max))
        for j in range(n):
            C[i,j] =  4.0 / (n-1)**2 * np.sum(B_x0 * (np.sin(u[i] * np.pi * y[:,None] / y_max) * (np.sin(u[j] * np.pi * z / z_max))))
    
    l[0,0] = 1
    
    r = np.sqrt(l - alpha**2)
    
    print 'calculating B'
    
    for i in range(n):
        for j in range(n):
            print j
            start = time()
            sincos = np.sin(np.pi * u[:,None] * y[i] / y_max) * (u * np.cos(np.pi * u * z[j] / z_max))
            cossin = (u[:,None] * np.cos(np.pi * u[:,None] * y[i] / y_max)) * (np.sin(np.pi * u * z[j] / z_max))
            temp_x = C * (np.sin(np.pi * u[:,None] * y[i] / y_max) * (np.sin(np.pi * u * z[j] / z_max)))
            temp_y = C / l * (alpha * np.pi / z_max * sincos - r * np.pi / y_max * cossin)
            temp_z = C / l * (alpha * np.pi / y_max * cossin + r * np.pi / z_max * sincos)
            
            Bx[:,i,j] = np.sum(np.sum(temp_x * np.exp(-r * x[:,None,None]),1),1)
            By[:,i,j] = np.sum(np.sum(temp_y * np.exp(-r * x[:,None,None]),1),1)
            Bz[:,i,j] = np.sum(np.sum(temp_z * np.exp(-r * x[:,None,None]),1),1)
            print time() - start

if __name__ == '__main__':
    fname = 'large.dat'
    main(fname)


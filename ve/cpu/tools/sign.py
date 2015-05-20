#!/usr/bin/env python
import numpy as np
import bohrium as bh

def thesign(z):
    bh.flush()
    x = np.real(z)
    xp = x*x
    del x
    y = np.imag(z)
    yp = y*y
    del y
    xpy = xp+yp
    del xp
    del yp
    xpy_sqrt = np.sqrt(xpy)
    del xpy
    z0 = z == 0
    sqrt_z0 = xpy_sqrt + z0
    del z0
    del xpy_sqrt
    out = z / sqrt_z0
    del sqrt_z0
    bh.flush()

    return out 

def sign_is(z):
    bh.flush()
    x = np.real(z)
    y = np.imag(z)
    out =  z / (np.sqrt(x*x+y*y)+(z==0))
    bh.flush()
    return out

z = np.asarray([0+1j,0-1j,0+0j,1+0j,1+1j,1-1j,0,1+0j,-1+1j,-1-1j,-1+0j], dtype=np.complex64)
print(thesign(z))
#print(sign_is(z))
print(np.sign(z))
#print(np.sign(w))
#print(np.sign(x))


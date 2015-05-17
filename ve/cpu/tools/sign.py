#!/usr/bin/env python
import numpy as np

def thesign(z):
    x = np.real(z)
    x_prod = x * x
    del x

    y = np.imag(z)
    y_prod = y * y
    del y
    xy = x_prod + y_prod
    del x_prod
    del y_prod
    xy_sqrt = np.sqrt(xy)
    
    ones = np.ones(z.shape, dtype=z.dtype)
    divisor = np.maximum(ones, xy_sqrt)

    return z/divisor

z = np.asarray([0+1j,0-1j,0+0j,1+0j,1+1j,1-1j,0,1+0j,-1+1j,-1-1j,-1+0j], dtype=np.complex64)
print(z)
print thesign(z)
#print(np.sign(x))


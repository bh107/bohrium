#!/usr/bin/env python
import numpy as np

def thesign(z):
    x = np.real(z)
    x_prod = x * x

    y = np.imag(z)
    y_prod = y * y
    xy = x_prod + y_prod
    xy_sqrt = np.sqrt(xy)
    
    z_zero = z == 0
    divisor = xy_sqrt + z_zero
    print("z", z)
    print("x", x)
    print("y", y)
    print("x*x", x_prod)
    print("y*y", y_prod)
    print("x+y", xy)
    print("sqrt", xy_sqrt)
    print("z_zero", z_zero)
    print("divisor", divisor)

    return z/divisor

z = np.asarray([0+1j,0-1j,0+0j,1+0j,1+1j,1-1j,0,1+0j,-1+1j,-1-1j,-1+0j], dtype=np.complex64)
w = np.arange(-10, 10, 1, dtype=np.float64)
x = np.arange(-10, 10, 1, dtype=np.int64)
#print(thesign(z))
print(np.sign(z))
print(np.sign(w))
print(np.sign(x))


#!/usr/bin/env python
import numpy as np
import cphvbnumpy as cnp
import time
import sys

def dim1_aca(size, cphvb):

    x = np.array([1]*size,  dtype=np.float64)
    y = 5
    z = np.empty((size),    dtype=np.float64)

    if cphvb:
        cnp.handle_array( x )
        cnp.handle_array( z )

    return x, y, z

def dim1_aaa(size, cphvb):

    x = np.array([1]*size,  dtype=np.float64)
    y = np.array([1]*size,  dtype=np.float64)
    z = np.empty((size),    dtype=np.float64)

    if cphvb:
        cnp.handle_array( x )
        cnp.handle_array( y )
        cnp.handle_array( z )
   
    return x, y, z

def dim2_aaa(size, cphvb):

    x = np.array([range(1,size+1)]*size, dtype=np.float64)
    y = np.array([range(1,size+1)]*size, dtype=np.float64)
    z = np.empty((size,size), dtype=np.float32)

    if cphvb:
        cnp.handle_array( x )
        cnp.handle_array( y )
        cnp.handle_array( z )

    return x, y, z

def dim3_aaa(size, cphvb):

    x = np.array([[range(1,size+1)]*size]*20, dtype=np.float64)
    y = np.array([[range(1,size+1)]*size]*20, dtype=np.float64)
    z = np.empty((20,size,size), dtype=np.float32)
    print x.shape
    if cphvb:
        cnp.handle_array( x )
        cnp.handle_array( y )
        cnp.handle_array( z )

    return x, y, z

def main():

    size    = 1024
    cphvb   = True

    try:
        cphvb = int(sys.argv[1])
    except:
        pass

    try:
        size = int(sys.argv[2])
    except:
        pass

    x, y, z = dim1_aaa( size, cphvb )
    #x, y, z = dim1_arr_con_arr( size, cphvb )
    #x, y, z = dim2_aaa( size, cphvb )
    #x, y, z = dim3_aaa( size, cphvb )

    start = time.time()
    np.add( x, y, z )
    print time.time() - start

if __name__ == "__main__":
    main()

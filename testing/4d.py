from pprint import pprint as pp
import numpy as np
import cphvbnumpy as cnp
import time
import sys

CPHVB   = True
size    = 3

try:
    CPHVB   = int(sys.argv[1])
    size    = int(sys.argv[2])
except:
    pass

def main():

    x = np.array([[[[3]*size]*size]*size]*size, dtype=np.float64)
    y = np.array([[[[6]*size]*size]*size]*size, dtype=np.float64)
    z = np.empty((size,size,size,size), dtype=np.float64)

    if CPHVB:
        cnp.handle_array( x ) 
        cnp.handle_array( y ) 
        cnp.handle_array( z )
 
    start = time.time() 
    np.add( x, y, z )
    print time.time()-start

if __name__ == "__main__":
    main()

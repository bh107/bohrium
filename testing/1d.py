from pprint import pprint as pp
import numpy as np
import cphvbnumpy as cnp
import time
import sys

CPHVB   = True
size    = 1024

try:
    CPHVB   = int(sys.argv[1])
    size    = int(sys.argv[2])
except:
    pass

def main():

    a = np.array([1]*size, dtype=np.float64)
    b = np.array([1]*size, dtype=np.float64)

    x = a[1:-1]
    y = b[0:-2]

    if CPHVB:
        cnp.handle_array( x ) 
        cnp.handle_array( y ) 
 
    start = time.time() 
    print np.add( x, y )
    print time.time()-start

if __name__ == "__main__":
    main()

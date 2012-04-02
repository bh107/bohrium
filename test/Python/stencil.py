from pprint import pprint as pp
import numpy as np
import cphvbnumpy as cnp
import time
import sys

CPHVB   = True
#size    = 1024
size    = 4

try:
    CPHVB   = int(sys.argv[1])
    size    = int(sys.argv[2])
except:
    pass

def main():

    x   = np.array([1]*size, dtype=np.float64)

    o1   = x[1::2]
    o2   = x[1::2]
    o3   = x[1::2]

    e1   = x[::2]
    e2   = x[::2]
    e3   = x[::2]

    if CPHVB:
        cnp.handle_array( x ) 
        cnp.handle_array( o1 ) 
        cnp.handle_array( o2 ) 
        cnp.handle_array( o3 ) 
        cnp.handle_array( e1 ) 
        cnp.handle_array( e2 ) 
        cnp.handle_array( e3 ) 
 
    start = time.time()

    np.add( o2, o3, o1 )
    np.subtract( o2, e1, e1 )

    print o1, e1

    print time.time()-start

if __name__ == "__main__":
    main()

from pprint import pprint as pp
import numpy as np
import cphvbnumpy as cnp

from time import time
import sys

CPHVB   = True
dim = [3, 3]

try:
    CPHVB   = int(sys.argv[1])
except:
    pass

def main():

    (w, h) = dim

    x = np.array([range(1, w+1)]*h, dtype=np.float32)
    y = np.array([range(1, w+1)]*h, dtype=np.float32)
    z = np.empty((h, w), dtype=np.float32)

    if CPHVB:
        cnp.handle_array( x )
        cnp.handle_array( y )
        cnp.handle_array( z )

    start = time() 
    np.add(x,y,z)
    print time() - start

if __name__ == "__main__":
    main()

# BH_FREE a4[0:1:1]
# BH_ADD a15[0:10:10,0:10:1] a1[1:11:12,1:11:1] a1[0:10:12,1:11:1]
# BH_ADD a23[0:10:10,0:10:1] a15[0:10:10,0:10:1] a1[2:12:12,1:11:1]
# BH_FREE a15[0:100:1]
# BH_ADD a207[0:10:10,0:10:1] a23[0:10:10,0:10:1] a1[1:11:12,2:12:1]
# BH_FREE a23[0:100:1]
# BH_ADD a26[0:10:10,0:10:1] a207[0:10:10,0:10:1] a1[1:11:12,0:10:1]
# BH_FREE a207[0:100:1]
# BH_MULTIPLY a13[0:10:10,0:10:1] 2.00000000000000011e-01 a26[0:10:10,0:10:1]
# BH_FREE a26[0:100:1]
# BH_FREE a148[0:100:1]
# BH_SUBTRACT a20[0:10:10,0:10:1] a13[0:10:10,0:10:1] a1[1:11:12,1:11:1]
# BH_ABSOLUTE a147[0:10:10,0:10:1] a20[0:10:10,0:10:1]
# BH_FREE a20[0:100:1]
# BH_ADD_REDUCE a17[0:10:1] a147[0:10:10,0:10:1] 1
# BH_ADD_REDUCE a28[0:1:1] a17[0:10:1] 0
# BH_FREE a17[0:10:1]
# BH_FREE a147[0:100:1]
# BH_FREE a2[0:1:1]
# BH_IDENTITY a1[1:11:12,1:11:1] a13[0:10:10,0:10:1]
# BH_GREATER a12[0:1:1] a28[0:1:1] 4.20000000000000000e+01

# Python version of the Heat Equation using Bohrium
import numpy as np

np.set_printoptions(linewidth=200)

def heat2d(height, width, epsilon=42):
    grid = np.zeros((height+2, width+2), dtype=np.float64)
    grid[:,0]  = -273.15
    grid[:,-1] = -273.15
    grid[-1,:] = -273.15
    grid[0,:]  = 40.0

    center = grid[1:-1,1:-1]
    north  = grid[:-2,1:-1]
    east   = grid[1:-1,2:]
    west   = grid[1:-1,:-2]
    south  = grid[2:,1:-1]

    delta = epsilon+1
    while delta > epsilon:
        tmp = 0.2 * (center + north + south + east + west)
        delta = np.sum(np.abs(tmp - center))
        center[:] = tmp

    return center

import sys
size = int(sys.argv[1])
result = heat2d(size, size)
try:
    np.flush()
except:
    pass
finally:
    print result[5][5]

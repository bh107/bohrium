import numpy as np
import matplotlib.pyplot as plt
import time
import sys

DEBUG = False

d = int(sys.argv[1]) #CUDA
S = int(sys.argv[2]) #Size of Model
I = int(sys.argv[3]) #Number of iterations

H = S 
W = S

full = np.zeros((W+2,H+2), dtype=np.float32, dist=d)

cells = full[1:W+1,1:H+1]
up = full[1:W+1, 0:H]
left = full[0:W, 1:H+1]
right = full[2:W+2, 1:H+1]
down = full[1:W+1, 2:H+2]

full[:,0] += -273.15
full[:,-1] += -273.15
full[0,:] += 40.0   
full[-1,:] += -273.13

top = 40
bot = -273

work = np.zeros((W,H), dtype=np.float32, dist=d)

t1 = time.time()

for i in range(I):
  np.add(cells,0.0,work)
  work += up
  work += left
  work += right
  work += down
  work *= 0.2
  np.add(work,0.0,cells)
#  np.core.multiarray.evalflush()  

t2 = time.time()

if DEBUG:
  plt.imshow(work)
  plt.show()

print d, " ", S, " ", I, " ", t2-t1





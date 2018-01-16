# Python version of the Heat Equation using Bohrium
import bohrium as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

fig = plt.figure()
plt.xticks([])
plt.yticks([])

def heat2d(height, width, epsilon=42):
    grid = np.zeros((height+2, width+2), dtype=np.float64)
    grid[:,0]  = np.float64(-273.15)
    grid[:,-1] = np.float64(-273.15)
    grid[-1,:] = np.float64(-273.15)
    grid[0,:]  = np.float64(40.0)

    center = grid[1:-1,1:-1]
    north  = grid[0:-2,1:-1]
    east   = grid[1:-1,2:]
    west   = grid[1:-1,0:-2]
    south  = grid[2:,1:-1]

    delta  = epsilon+1
    i = 0
    # plt.imshow(grid.copy2numpy(), cmap="coolwarm")
    # plt.draw()
    while delta > epsilon:
        tmp = 0.2 * (center + north + south + east + west)
        delta = np.sum(np.abs(tmp - center))
        grid[1:-1,1:-1] = tmp
        if i % 100 == 0:
            pass
            # plt.imshow(center.copy2numpy(), cmap="coolwarm")
            # plt.draw()
            # plt.pause(0.000000000001)
        i += 1

    return center

result = heat2d(100, 100)
# plt.show()
# plt.imshow(result.copy2numpy(), cmap="coolwarm")
# plt.show()

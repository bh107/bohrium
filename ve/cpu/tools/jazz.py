from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def make_cube():
    """ A Cube consists of a bunch of planes..."""

    planes = {
        "top"    : ( [[0,1],[0,1]], [[0,0],[1,1]], [[1,1],[1,1]] ),
        "bottom" : ( [[0,1],[0,1]], [[0,0],[1,1]], [[0,0],[0,0]] ),
        "left"   : ( [[0,0],[0,0]], [[0,1],[0,1]], [[0,0],[1,1]] ),
        "right"  : ( [[1,1],[1,1]], [[0,1],[0,1]], [[0,0],[1,1]] ),
        "front"  : ( [[0,1],[0,1]], [[0,0],[0,0]], [[0,0],[1,1]] ),
        "back"   : ( [[0,1],[0,1]], [[1,1],[1,1]], [[0,0],[1,1]] )
    }
    return planes

cube = make_cube()

A = np.ones((2,3,4))


highlight = np.zeros(A.shape)
highlight[1,1,:] = 1

for space in xrange(0, A.shape[0]):
    for column in xrange(0, A.shape[1]):
        for row in xrange(0, A.shape[2]):
            alpha = 0.01
            if highlight[space,column,row] == 1:
                alpha = 1
            for side in cube:
                (Xs, Ys, Zs) = (
                    np.asarray(cube[side][0])+space,
                    np.asarray(cube[side][1])+row,
                    np.asarray(cube[side][2])+column
                )
                ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, alpha=alpha)


highest = 0                     # Make it look cubic
for size in A.shape:
    if size > highest:
        highest = size
ax.set_xlim((0,highest))
ax.set_ylim((0,highest))
ax.set_zlim((0,highest))

ax.set_xlabel('Space')          # Meant to visualize ROW-MAJOR ordering 
ax.set_ylabel('Rows')
ax.set_zlabel('Column')

plt.show()

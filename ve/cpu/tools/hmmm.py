from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['grid.alpha'] = 0.0

def make_cube():
    """ A Cube consists of a bunch of planes..."""

    planes = {
        "top"    : np.asarray( [[[0,1],[0,1]], [[0,0],[1,1]], [[1,1],[1,1]]] ),
        "bottom" : np.asarray( [[[0,1],[0,1]], [[0,0],[1,1]], [[0,0],[0,0]]] ),
        "left"   : np.asarray( [[[0,0],[0,0]], [[0,1],[0,1]], [[0,0],[1,1]]] ),
        "right"  : np.asarray( [[[1,1],[1,1]], [[0,1],[0,1]], [[0,0],[1,1]]] ),
        "front"  : np.asarray( [[[0,1],[0,1]], [[0,0],[0,0]], [[0,0],[1,1]]] ),
        "back"   : np.asarray( [[[0,1],[0,1]], [[1,1],[1,1]], [[0,0],[1,1]]] )
    }
    return planes

def skew(space, column, row, recipe=None):
    """Position of array elements in relation to each other."""

    if recipe is None:
        return (0,0,0)
    elif recipe == "layered_tight":
        return (space*0.1, space*0.5, -(space*0.5))
    elif recipe == "layered_loose":
        return (space*1.0, space*1.25, -(space*1.25))
    elif recipe == "layered":
        return (space*0.75, space*0.75, -(space*0.75))
    else:
        raise TypeError("Unknown recipe[%s]" % recipe)

class NDArrayPlotter(object):

    def __init__(self, shape, color="blue", alpha="0.6"):
        self.defaults = {
            "color": color,
            "alpha": alpha,
            "shape": shape
        }
        self.set_shape(shape)
        self.set_color(color)
        self.set_alpha(alpha)

    def set_shape(self, shape):
        self.shape = shape

    def set_color(self, color):
        self.colors = np.zeros(self.shape, dtype=('a10'))
        self.colors[:] = color

    def set_alpha(self, alpha):
        self.alphas = np.empty(self.shape, dtype=np.float32)
        self.alphas[:] = alpha

    def render(self, ary, highlight=None, color=None, alpha=None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cube = make_cube()

        scale = (0.01, 0.75, 0.75)
        for space in xrange(0, ary.shape[0]):
            for column in xrange(0, ary.shape[1]):
                for row in xrange(0, ary.shape[2]):

                    # Extract settings that apply to all sides of the cube
                    alpha = self.alphas[space,column,row]
                    color = self.colors[space,column,row]

                    if highlight and highlight[space,column,row] == 1:
                        alpha = 1

                    relative_pos = skew(space, column, row, 'layered')

                    for side in cube:
                        (Xs, Ys, Zs) = (
                            scale[0]*(cube[side][0] + space  ) +relative_pos[0],
                            scale[1]*(cube[side][1] + row    ) +relative_pos[1],
                            scale[2]*(cube[side][2] + column ) +relative_pos[2]
                        )
                        ax.plot_surface(
                            Xs, Ys, Zs,
                            rstride=1, cstride=1,
                            alpha=alpha,
                            color=color
                        )

        highest = 0                         # Make it look cubic
        for size in ary.shape:
            if size > highest:
                highest = size
        ax.set_xlim((0,highest))
        ax.set_ylim((0,highest))
        ax.set_zlim((0,highest))


        ax.set_xlabel('Third dimension' )   # Meant to visualize ROW-MAJOR ordering 
        ax.set_ylabel('Row(s)')
        ax.set_zlabel('Column(s)')

        #plt.axis('off')    # This also removes the axis labels... i want those...
        #ax.set_axis_off()  # this removes too much (also the labels)

        # So I try this instead...
        ax.set_xticks([])          # removes the ticks... great now the rest of it
        ax.set_yticks([])
        ax.set_zticks([])
        #ax.grid(False)             # this does nothing....
        #ax.set_frame_on(False)     # this does nothing....
        plt.show()

def main():
    
    subject = np.ones((5,5,5))

    #highlight = np.zeros(subject.shape) # Highlight a row
    #highlight[ 0,  :, :] = 1
    #highlight[-1,  :, :] = 1
    #highlight[ :,  0, :] = 1
    #highlight[ :, -1, :] = 1
    #highlight[ :,  :, 0] = 1
    #highlight[ :,  :,-1] = 1


    plotter = NDArrayPlotter(subject.shape)

    colors = plotter.colors
    colors[:] = "#00FF00"
    colors[ 0,  :, :] = "#FF0000"
    colors[-1,  :, :] = "#FF0000"
    colors[ :,  0, :] = "#FF0000"
    colors[ :, -1, :] = "#FF0000"
    colors[ :,  :, 0] = "#FF0000"
    colors[ :,  :,-1] = "#FF0000"

    plotter.render(subject)

if __name__ == "__main__":
    main()

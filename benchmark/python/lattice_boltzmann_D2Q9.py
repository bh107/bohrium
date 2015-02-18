from __future__ import print_function
"""
The Lattice Boltzmann Methods D2Q9
------------

Channel flow past a cylindrical obstacle, using a LB method
Copyright (C) 2006 Jonas Latt
Address: Rue General Dufour 24,  1211 Geneva 4, Switzerland
E-mail: Jonas.Latt@cui.unige.ch
"""
import threading
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

# D2Q9 Lattice constants
t       = [4/9., 1/9.,1/9.,1/9.,1/9., 1/36.,1/36.,1/36.,1/36.]
cx      = [  0,   1,  0, -1,  0,    1,  -1,  -1,   1]
cy      = [  0,   0,  1,  0, -1,    1,   1,  -1,  -1]
opp     = [ 0,   3,  4,  1,  2,    7,   8,   5,   6]
uMax    = 0.02 # maximum velocity of Poiseuille inflow
Re      = 100  # Reynolds number

def cylinder(height, width, obstacle=True, dtype=np.float32):
    assert(height > 2)
    assert(width > 2)

    lx = width
    ly = height
    state   = {"lx":lx, "ly":ly}
    obst_x  = lx/5.+1               # position of the cylinder; (exact
    obst_y  = ly/2.+0               # y-symmetry is avoided)
    obst_r  = ly/10.+1              # radius of the cylinder

    nu     = uMax * 2.*obst_r / Re  # kinematic viscosity
    omega  = 1. / (3*nu+1./2.)      # relaxation parameter

    col = np.arange(1, ly - 1)

    # We need some of the constants as 3D numpy arrays
    t_3d    = np.asarray(t)[:, np.newaxis, np.newaxis]
    cx_3d   = np.asarray(cx)[:, np.newaxis, np.newaxis]
    cy_3d   = np.asarray(cy)[:, np.newaxis, np.newaxis]

    # Place obstacle
    if obstacle:
      y,x = np.meshgrid(np.arange(ly),np.arange(lx))
      obst = (x-obst_x)**2 + (y-obst_y)**2 <= obst_r**2
      obst[:,0] = obst[:,ly-1] = 1
      bbRegion = np.asarray(obst, dtype = np.bool)
      #bbRegion.bohrium = bohrium
    else:
      bbRegion = None

    # Initial condition: (rho=0, u=0) ==> fIn[i] = t[i]
    fIn = t_3d.repeat(lx, axis = 1).repeat(ly, axis = 2)

    #fIn.bohrium = bohrium
    #cx_3d.bohrium = bohrium
    #cy_3d.bohrium = bohrium
    #col.bohrium = bohrium
    state['fIn'] = fIn
    state['cx_3d'] = cx_3d
    state['cy_3d'] = cy_3d
    state['col'] = col
    state['omega'] = omega
    state['bbRegion'] = bbRegion

    return state

def solve(state, iterations, viz=None):

    #load the state
    ly = state['ly']
    lx = state['lx']
    fIn = state['fIn']
    cx_3d = state['cx_3d']
    cy_3d = state['cy_3d']
    col = state['col']
    omega = state['omega']
    bbRegion = state['bbRegion']

    # Main loop (time cycles)
    for cycle in xrange(iterations):

      # Macroscopic variables
      rho = np.sum(fIn, axis = 0)
      ux = np.sum(cx_3d * fIn, axis = 0) / rho
      uy = np.sum(cy_3d * fIn, axis = 0) / rho

      # Macroscopic (Dirichlet) boundary conditions

      # Inlet: Poisseuille profile
      L = ly-2.0
      y = col-0.5
      ux[0, 1:ly-1] = 4 * uMax / (L ** 2) * (y * L - y ** 2)
      uy[0, 1:ly-1] = 0

      #Using a loop instead of "index of indexes"
      # t1 = fIn[[0, 2, 4], 0, 1:ly-1].sum(axis = 0)
      #t1 = np.zeros(ly-2, bohrium=fIn.bohrium)
      t1 = np.zeros(ly-2)
      for i in [0, 2, 4]:
        t1 += fIn[i, 0, 1:ly-1]

      #Using a loop instead of "index of indexes"
      # t2 = 2 * fIn[[3, 6, 7], 0, 1:ly-1].sum(axis = 0)
      #t2 = np.zeros(ly-2, bohrium=fIn.bohrium)
      t2 = np.zeros(ly-2)
      for i in [3, 6, 7]:
        t2 += fIn[i, 0, 1:ly-1]
      t2 *= 2

      rho[0, 1:ly-1] = 1 / (1 - ux[0, 1:ly-1]) * (t1 + t2)

      # Outlet: Zero gradient on rho/ux
      rho[lx - 1, 1:ly-1] = 4.0 / 3 * rho[lx - 2, 1:ly-1] - \
                         1.0 / 3 * rho[lx - 3, 1:ly-1]
      uy[lx - 1, 1:ly-1] = 0
      ux[lx - 1, 1:ly-1] = 4.0 / 3 * ux[lx - 2, 1:ly-1] - \
                        1.0 / 3 * ux[lx - 3, 1:ly-1]

      #fEq = np.zeros((9, lx, ly), bohrium=fIn.bohrium)
      fEq = np.zeros((9, lx, ly))
      #fOut = np.zeros((9, lx, ly), bohrium=fIn.bohrium)
      fOut = np.zeros((9, lx, ly))
      for i in xrange(0, 9):
          cu = 3 * (cx[i] * ux + cy[i] * uy)
          fEq[i] = rho * t[i] * (1 + cu + 0.5 * cu ** 2 - \
                          1.5 * (ux ** 2 + uy ** 2))
          fOut[i] = fIn[i] - omega * (fIn[i] - fEq[i])

      # Microscopic boundary conditions
      for i in xrange(0, 9):
          # Left boundary:
          fOut[i, 0, 1:ly-1] = fEq[i,0,1:ly-1] + 18 * t[i] * cx[i] * cy[i] * \
             (fIn[7,0,1:ly-1] - fIn[6,0,1:ly-1] - fEq[7,0,1:ly-1] + fEq[6,0,1:ly-1])
          # Right boundary:
          fOut[i,lx-1,1:ly-1] = fEq[i,lx-1,1:ly-1] + 18 * t[i] * cx[i] * cy[i] * \
             (fIn[5,lx-1,1:ly-1] - fIn[8,lx-1,1:ly-1] - \
              fEq[5,lx-1,1:ly-1] + fEq[8,lx-1,1:ly-1])
          # Bounce back region:
          #fOut[i,bbRegion] = fIn[opp[i],bbRegion]
          #Using a explict mask
          if bbRegion is not None:
              masked = fIn[opp[i]].copy() * bbRegion
              fOut[i] = fOut[i] * ~bbRegion + masked


      # Streaming step
      for i in xrange(0,9):
          #fIn[i] = np.roll(np.roll(fOut[i], cx[i], axis = 0), cy[i], axis = 1)
          #Replacing the np.roll() call with:
          if cx[i] == 1:
              #t1 = np.empty_like(fOut[i], bohrium=fIn.bohrium)
              t1 = np.empty_like(fOut[i])
              t1[1:] = fOut[i][:-1]
              t1[0] = fOut[i][-1]
              fOut[i] = t1
          elif cx[i] == -1:
              #t1 = np.empty_like(fOut[i], bohrium=fIn.bohrium)
              t1 = np.empty_like(fOut[i])
              t1[:-1] = fOut[i][1:]
              t1[-1] = fOut[i][0]
              fOut[i] = t1
          if cy[i] == 1:
              #t1 = np.empty_like(fOut[i], bohrium=fIn.bohrium)
              t1 = np.empty_like(fOut[i])
              t1[:,1:] = fOut[i][:,:-1]
              t1[:,0] = fOut[i][:,-1]
              fIn[i] = t1
          elif cy[i] == -1:
              #t1 = np.empty_like(fOut[i], bohrium=fIn.bohrium)
              t1 = np.empty_like(fOut[i])
              t1[:,:-1] = fOut[i][:,1:]
              t1[:,-1] = fOut[i][:,0]
              fIn[i] = t1
          else:
              fIn[i] = fOut[i]

      if viz and not cycle % 10:
          viz['axes1'].clear()
          viz['axes2'].clear()
          viz['axes3'].clear()
          viz['axes1'].imshow(fIn[4].T.copy())
          viz['axes2'].imshow(ux.T.copy())
          viz['axes3'].imshow(uy.T.copy())
          viz['canvas'].show()

def setup_viz(state):
    """Setup visualization."""

    import matplotlib
    import matplotlib.figure
    import matplotlib.backends.backend_tkagg
    import Tkinter
    import threading

    viz = {}

    # Initialise Tk ...
    figure = matplotlib.figure.Figure()
    figure.set_size_inches((8, 6))

    viz['axes1'] = figure.add_subplot(311)
    viz['axes2'] = figure.add_subplot(312)
    viz['axes3'] = figure.add_subplot(313)

    viz['tk'] = Tkinter.Tk()
    viz['canvas'] = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
        figure,
        master=viz['tk']
    )
    viz['canvas'].get_tk_widget().pack(expand = True, fill = Tkinter.BOTH)

    return viz 

def main():
    B = util.Benchmark()
    H = B.size[0]
    W = B.size[1]
    I = B.size[2]

    state = cylinder(H, W, obstacle=False)

    viz = None
    if B.verbose:
        viz = setup_viz(state)

    B.start()
    if viz:
        thread = threading.Thread(target = solve, args=(state, I, viz))
        thread.start()
        viz['tk'].mainloop()   
    else:
        R = solve(state, I, viz)
    B.stop()
    B.pprint()

    if B.outputfn:
        B.tofile(B.outputfn, {'res': np.array(R)})

if __name__ == "__main__":
    main()

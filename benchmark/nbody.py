import numpy as np
import util
import cphvbnumpy

B = util.Benchmark()
n = B.size[0]
k = B.size[1]

G = 1     #Gravitational constant
dT = 0.01 #Time increment

M   = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis] + 0.1
MT  = np.random.random(n, cphvb=B.cphvb)[np.newaxis,:] + 0.1
Px  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]
PxT = np.random.random(n, cphvb=B.cphvb)[np.newaxis,:]
Py  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]
PyT = np.random.random(n, cphvb=B.cphvb)[np.newaxis,:]
Pz  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]
PzT = np.random.random(n, cphvb=B.cphvb)[np.newaxis,:]
Vx  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]
Vy  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]
Vz  = np.random.random(n, cphvb=B.cphvb)[:,np.newaxis]

OnesCol = np.empty((n,1), dtype=float, dist=B.cphvb)
OnesCol[:] = 1.0
OnesRow = np.empty((1,n), dtype=float, dist=B.cphvb)
OnesRow[:] = 1.0
Identity= np.array(np.diag([1]*n), dtype=float)
if B.cphvb:
    cphvbnumpy.handle_array(Identity)

B.start()
for i in xrange(k):
    print "np.dot is not supported in CPHVB "
    #distance between all pairs of objects
    Fx = np.dot(OnesCol, PxT) - np.dot(Px, OnesRow)
    Fy = np.dot(OnesCol, PyT) - np.dot(Py, OnesRow)
    Fz = np.dot(OnesCol, PzT) - np.dot(Pz, OnesRow)
    if B.cphvb:
        cphvbnumpy.handle_array(Fx)
        cphvbnumpy.handle_array(Fy)
        cphvbnumpy.handle_array(Fz)

    Dsq = Fx * Fx
    Dsq += Fy * Fy
    Dsq += Fz * Fz
    Dsq += Identity
    D = np.sqrt(Dsq)

    #mutual forces between all pairs of objects
    F = np.dot(M, MT)
    F *= G
    F /= Dsq
    #F = F - diag(diag(F))#set 'self attraction' to 0
    Fx /= D
    Fx *= F
    Fy /= D
    Fy *= F
    Fz /= D
    Fz *= F

    #net force on each body
    Fnet_x = np.add.reduce(Fx,1)
    Fnet_y = np.add.reduce(Fy,1)
    Fnet_z = np.add.reduce(Fz,1)

    Fnet_x = Fnet_x[:,np.newaxis] * dT
    Fnet_y = Fnet_y[:,np.newaxis] * dT
    Fnet_z = Fnet_z[:,np.newaxis] * dT

    #change in velocity:
    Vx += Fnet_x / M
    Vy += Fnet_y / M
    Vz += Fnet_z / M

    #change in position
    Px += Vx * dT
    Py += Vy * dT
    Pz += Vz * dT
B.stop()
B.pprint()




"""Paper version:
    #distance between all pairs of objects
    Fx = dot(OnesCol, PxT) - dot(Px, OnesRow)
    Fy = dot(OnesCol, PyT) - dot(Py, OnesRow)
    Fz = dot(OnesCol, PzT) - dot(Pz, OnesRow)

    Dsq = Fx * Fx + Fy * Fy + Fx * Fz #+ Identity
    D = sqrt(Dsq)

    #mutual forces between all pairs of objects
    F = G * dot(M, MT) / Dsq

    #F = F - diag(diag(F))#set 'self attraction' to 0
    Fx = (Fx / D) * F
    Fy = (Fy / D) * F
    Fz = (Fz / D) * F

    #net force on each body
    Fnet_x = add.reduce(Fx,1)
    Fnet_x = add.reduce(Fy,1)
    Fnet_x = add.reduce(Fz,1)

    #change in velocity:
    Vx += Fnet_x[:,newaxis] * dT / M
    Vy += Fnet_y[:,newaxis] * dT / M
    Vz += Fnet_z[:,newaxis] * dT / M

    #change in position
    Px += Vx * dT
    Py += Vy * dT
    Pz += Vz * dT
"""

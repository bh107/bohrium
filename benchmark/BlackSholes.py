#!/usr/bin/env python
import numpy as np
import sys
import time

DIST=int(sys.argv[1])
N=int(sys.argv[2])
year=float(sys.argv[3])


def CND(X):
    mask = np.empty(N, dtype=np.bool, dist=DIST)
    _mask = np.empty(N, dtype=np.bool, dist=DIST)
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, \
                        -1.821255978, 1.330274429)
    L = np.abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / np.sqrt(2*np.pi)*np.exp2((-L*L/2.)*np.log2(np.e)) * \
        (a1*K + a2*(K*K) + a3*(K*K*K) + a4*(K*K*K*K) + a5*(K*K*K*K*K))

    np.less(X,0.0,mask)
    np.greater_equal(X, 0.0,_mask)
    #np.multiply(w,_mask,w)
    w = w * _mask + (1.0-w)*mask
    #w = w * _mask
    
    return w

# Black Sholes Function
def BlackSholes(CallPutFlag,S,X,T,r,v):
    d1 = (np.log2(S/X)/np.log2(np.e)+(r+v*v/2.)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='c':
        return S*CND(d1)-X*np.exp2((-r*T)*np.log2(np.e))*CND(d2)
    else:
        return X*np.exp2((-r*T)*np.log2(np.e))*CND(-d2)-S*CND(-d1)


S = np.empty((N), dtype=np.float32, dist=DIST)
R = np.empty(1, dtype=np.float32, dist=DIST)
if DIST:
    np.core.multiarray.random(S)
else:
    S = np.random.rand(N).astype(np.float32)


np.core.multiarray.evalflush()
S = S*4.0-2.0 + 60.0 #Price is 58-62
#S = S + 60.0 #Price is 58-62
print S
day=1.0/365.0
X=65.0
r=0.08
v=0.3
T=day

stime = time.time()
while T < year:
    np.add.reduce(BlackSholes('c', S, X, T, r, v),out=R)
    res = R[0]/N
    T+=day
#    print "res: ", res
np.core.multiarray.evalflush()
print DIST, " ", N, " ", int(year*365), " ",  time.time() - stime,

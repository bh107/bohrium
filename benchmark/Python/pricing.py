from __future__ import print_function
"""
#
# NumPy version of Rolf Poulsens code for pricing american options.
#
# R-code source: http://www.math.ku.dk/~rolf/FAMOES/UsualBinomAmrPut.R 
#
source("UsualBinomAmrPut.R")

S0<-100;r<-0.03; alpha<-0.07; sigma<-0.20;

expiry<-1; strike<-100

n<-expiry*252; dt<-expiry/n
u<-exp(alpha*dt+sigma*sqrt(dt)); d<-exp(alpha*dt-sigma*sqrt(dt))
R<-exp(r*dt)

q<-(R-d)/(u-d)

put<-matrix(0,nrow=(n+1),ncol=(n+1))

put[,n+1]<-pmax(strike-S0*u^(0:n)*d^(n:0),0)

for (i in n:1) {
  for (j in 1:i){
     S<-S0*u^(j-1)* d^(i-j) 
     put[j,i]<-max(max(strike-S,0),(q*put[j+1,i+1]+(1-q)*put[j,i+1])/R) 
   }
}

print(put[1,1])
"""
import math
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def main():
    S0      = 100.0         # Pricing parameters
    r       = 0.03
    alpha   = 0.07
    sigma   = 0.20
    expiry  = 1.0
    strike  = 100.0

    n   = int(expiry * 252)
    dt  = expiry / n
    u   = math.exp( alpha*dt + sigma*math.sqrt(dt) )
    d   = math.exp( alpha*dt - sigma*math.sqrt(dt) )
    R   = math.exp( r*dt )
    q   = (R-d)/(u-d)

    asc     = np.array( range(0, n+1) )
    desc    = np.array( range(n,-1,-1) )

    B = util.Benchmark()
    B.start()

    put         = np.zeros( (n+1, n+1) )
    put[:,n]    = np.maximum( strike - S0* (u**asc) * (d**desc), 0 )

    for i in xrange(n-1, -1, -1):
        for j in xrange(0, i+1):
            S = S0*u*(j-1)* (d**(i-j))
            put[j,i] = max(max(strike-S, 0.0), (q*put[j+1,i+1]+(1-q) * put[j,i+1])/R)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()

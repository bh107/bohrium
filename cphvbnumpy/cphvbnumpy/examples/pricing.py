#!/usr/bin/env python
#
# NumPy version of Rolf Poulsens code for pricing american options.
#
# R-code source: http://www.math.ku.dk/~rolf/FAMOES/UsualBinomAmrPut.R 
#
"""
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
import cphvbnumpy as np

S0  = 100
r   = 0.03
alpha   = 0.07
sigma   = 0.20
expiry  = 1
strike  = 1

n   = expiry * 252
dt  = expiry / n
u   = np.exp( alpha*dt + sigma*np.sqrt(dt) )
d   = np.exp( alpha*dt - sigma*np.sqrt(dt) )
R   = np.exp( r*dt )
q   = (R-d)/(u-d)

put = np.array( (n+1, n+1) )
# put[,n+1] = ...

for i in xrange(0, n):
    for j in xrange(1, i):
        S = S0*u*(j-1)*d^(i-j)
        put[j][j] = max(max(strike-S,0), (q*put[j+1,][i+1]+(1-q)*put[j][i+1])/R)


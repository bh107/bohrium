import cphvbnumpy as np
import util

# Cumulative normal distribution
def CND(X):

    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = np.absolute(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / np.sqrt(2*np.pi)*np.exp(-L*L/2.) * \
        (a1*K + a2*(K**2) + a3*(K**3) + a4*(K**4) + a5*(K**5))

    mask    = X<0
    w       = w * ~mask + (1.0-w)*mask  # This probably leads to a copy/identity,
                                        # Since we have mixed input-types.
    return w

# Black Sholes Function
def BS(CallPutFlag,S,X,T,r,v):

    d1 = (np.log(S/X)+(r+v*v/2.)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='c':
        return S*CND(d1)-X*np.exp(-r*T)*CND(d2)
    else:
        return X*np.exp(-r*T)*CND(-d2)-S*CND(-d1)

def main():

    B       = util.Benchmark()
    N       = B.size[0]
    year    = B.size[1]

    S = np.random.random([N], cphvb=B.cphvb)
    S = S*4.0-2.0 + 60.0 # Price is 58-62

    X   = 65.0
    r   = 0.08
    v   = 0.3
    day = 1.0/year
    T   = day

    B.start()
    for t in xrange(year):
        np.sum(BS('c', S, X, T, r, v)) / N
        T += day
    B.stop()
    B.pprint()   

if __name__ == "__main__":
    main()

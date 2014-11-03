from __future__ import print_function
# Leif Andersen - A simple Approach to the pricing of Bermudan swaptions
# in the multifactor LIBOR market model - 1999 - Journal of computational finance.
# Replication of Table 1 p. 16 - European Payer Swaptions.
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

# Various global values

delta = 0.5             # Parameter values.
f_0   = 0.06
theta = 0.06

# Start dates for a series of swaptions.
Ts_all      = [[1,2,3],[2,3,4],[5,6,7,8,9],[10,12,14,16,18]]
# End dates for a series of swaptions.
Te_all      = [4,5,10,20]
# Parameter values for a series of swaptions.
lamb_all    = [0.2,0.2,0.15,0.1]

def main(verbose=False):
    """Set verbose=true to output the computed prices."""

    # Container for the value and std for the series of swaptions.
    swaptions = np.zeros((2,1))

    # Auxiliary function.
    def mu(F):
        tmp = lamb*(delta*F[1:,:])/(1+delta*F[1:,:]) # Andreasen style
        mu = np.zeros((tmp.shape))
        mu[0,:] +=tmp[0,:]
        for i in xrange(mu.shape[0]-1):
            mu[i+1,:] = mu[i,:] + tmp[i+1,:]
        return mu


    B = util.Benchmark()
    N = B.size[0]           # Number of paths.

    B.start()

    # Range over a number of independent products
    for i in xrange(len(Te_all)):
        Te = Te_all[i]
        lamb = lamb_all[i]
        Ts = Ts_all[i]
        for ts in Ts:

            time_structure = np.arange(0,ts+delta,delta)
            maturity_structure = np.arange(0,Te,delta)

            ############### MODEL #######################

            # Variance reduction technique - Antithetic Variates.
            eps_tmp = np.random.normal(loc = 0.0, scale = 1.0, size = ((len(time_structure)-1),N))
            eps = np.concatenate((eps_tmp,-eps_tmp), axis = 1)

            # Forward LIBOR rates for the construction of the spot measure.
            F_kk = np.zeros((len(time_structure),2*N))
            F_kk[0,:] += f_0

            # Plane kxN of simulated LIBOR rates.
            F_kN = np.ones((len(maturity_structure),2*N))*f_0

            # Simulations of the plane F_kN for each time step.
            for t in xrange(1,len(time_structure),1):
                F_kN_new = np.ones((len(maturity_structure)-t,2*N))
                F_kN_new = F_kN[1:,:]*np.exp(lamb*mu(F_kN)*delta-0.5*lamb*lamb*delta+lamb*eps[t-1,:]*np.sqrt(delta))
                F_kk[t,:] = F_kN_new[0,:]
                F_kN = F_kN_new

            ############### PRODUCT #####################

            # Value of zero coupon bonds.
            zcb = np.ones((int((Te-ts)/delta)+1,2*N))
            for j in xrange(len(zcb)-1):
                zcb[j+1,:] = zcb[j,:]/(1+delta*F_kN[j,:])

            # Swaption price at maturity.
            swap_Ts = np.maximum(1-zcb[-1,:]-theta*delta*np.sum(zcb[1:,:], axis = 0),0)

            # Spot measure used for discounting.
            B_Ts = np.ones((2*N))
            for j in xrange(int(ts/delta)):
                B_Ts *= (1+delta*F_kk[j,:])

            # Swaption price at time 0.
            swaption = swap_Ts/B_Ts

            # Save expected value in bps and std.
            swaptions = np.append(swaptions,[[np.average( (swaption[0:N] + swaption[N:])/2) *10000],\
                [np.std((swaption[0:N] + swaption[N:])/2)/np.sqrt(N)*10000]], axis=1)

    B.stop()
    B.pprint()

    if verbose:     # Print values.
        k=1
        for i in xrange(len(Te_all)):
            Te = Te_all[i]
            Ts = Ts_all[i]
            for j in xrange(len(Ts)):
                print("Ts %i" %Ts[j] + " Te %i " %Te + " price %.2f" % swaptions[0,k]\
                               + "(%.2f)" %swaptions[1,k])
                k +=1

if __name__ == "__main__":
    main()

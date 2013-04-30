import bohrium as np
import sys

def monte_carlo_pi(samples, iterations):

    s = np.zeros(1)
    for i in xrange(0, iterations):                     # sample
        x = np.random.random((samples), dtype=np.float64, bohrium=True)
        y = np.random.random((samples), dtype=np.float64, bohrium=True)

        m = np.sqrt(x*x + y*y)                          # model
        c = np.add.reduce((m<1.0).astype('float'))      # count

        s += c*4.0 / samples                            # approximate

    return s / (iterations)

if __name__ == "__main__":
    print 'Pi Approximation: ', monte_carlo_pi(int(sys.argv[1]),
                                               int(sys.argv[2]))

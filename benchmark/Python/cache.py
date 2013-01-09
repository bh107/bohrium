import cphvbnumpy as np
import util

def main():

    b = util.Benchmark()
    n = b.size[0]
    c = b.size[1]

    x = np.random.random([n], cphvb=b.cphvb)

    b.start()

    #for _ in xrange(i):
    #    for _ in xrange(c):
    #        #x = np.sin(np.sin( y ))
    #        x = (y+y+2)*5
    #x = 2*y
    #x[:]= y+y+2
    #x= y+y+2
    #x = np.sin(np.sin(np.sin(y)))
    for _ in xrange(c):
        x += np.sin(x)

    b.stop()
    b.pprint()   

if __name__ == "__main__":
    main()

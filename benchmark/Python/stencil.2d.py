import cphvbnumpy as numpy
import util
    
def main():

    b = util.Benchmark()
    n = b.size[0]
    m = b.size[1]
    i = b.size[2]

    b.start()

    raw = numpy.ones((n+4, m+4), cphvb=b.cphvb)

    data =   raw[2:-2, 2:-2]
    up2 =    raw[2:-2,  :-4]
    up1 =    raw[2:-2, 1:-3]
    down1 =  raw[2:-2, 3:-1]
    down2 =  raw[2:-2, 4:  ]
    left2 =  raw[ :-4, 2:-2]
    left1 =  raw[1:-3, 2:-2]
    right1 = raw[3:-1, 2:-2]
    right2 = raw[4:  , 2:-2]

    for _ in xrange(i):
        tmp = (data+left1+right1+left2+right2+up2+up1+down2+down1)/9
        data[:] = tmp

    b.stop()
    b.pprint()

if __name__ == "__main__":
    main()

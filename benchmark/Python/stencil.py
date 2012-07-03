import cphvbnumpy as numpy
import util

def onethree(n, m, i, b):

    raw = numpy.ones((n**m)+2)
    raw.cphvb = b.cphvb
    data =  raw[1:-1]
    left =  raw[ :-2]
    right = raw[2:  ]

    for _ in xrange(i):
        tmp = (data+left+right)/3
        data[:] = tmp
    raw.cphvb = False

def onefive(n, m, i, b):

    raw = numpy.ones(n**m+4)
    raw.cphvb = b.cphvb
    data =   raw[2:-2]
    left2 =  raw[ :-4]
    left1 =  raw[1:-3]
    right1 = raw[3:-1]
    right2 = raw[4:  ]

    for _ in xrange(i):
        tmp = (data+left1+right1+left2+right2)/5
        data[:] = tmp
    raw.cphvb = False

def twofive(n, m, i, b):

    raw = numpy.ones((n+2,n+2))
    raw.cphvb = b.cphvb
    data =  raw[1:-1, 1:-1]
    left =  raw[ :-2, 1:-1]
    right = raw[2:  , 1:-1]
    up =    raw[1:-1,  :-2]
    down =  raw[1:-1, 2:  ]

    for _ in xrange(i):
        tmp = (data+left+right+up+down)/5
        data[:] = tmp
    raw.cphvb = False

def twonine(n, m, i, b):

    raw = numpy.ones((n+4, m+4))
    raw.cphvb=b.cphvb

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
    raw.cphvb = False

def threeseven(n, m, i, b):

    raw = numpy.ones((n+2,n+2, n+2))
    raw.cphvb=b.cphvb
    data =  raw[1:-1, 1:-1, 1:-1]
    left =  raw[ :-2, 1:-1, 1:-1]
    right = raw[2:  , 1:-1, 1:-1]
    up =    raw[1:-1,  :-2, 1:-1]
    down =  raw[1:-1, 2:  , 1:-1]
    zin =   raw[1:-1, 1:-1,  :-2]
    zout =  raw[1:-1, 1:-1, 2:  ]

    for _ in xrange(i):
        tmp = (data+left+right+up+down+zin+zout)/7
        data[:] = tmp
    raw.cphvb = False

def threethirtheen(n, m, i, b):

    raw = numpy.ones((n+4, n+4, n+4))
    raw.cphvb=b.cphvb
    data =   raw[2:-2, 2:-2, 2:-2]
    up2 =    raw[2:-2,  :-4, 2:-2]
    up1 =    raw[2:-2, 1:-3, 2:-2]
    down1 =  raw[2:-2, 3:-1, 2:-2]
    down2 =  raw[2:-2, 4:  , 2:-2]
    left2 =  raw[ :-4, 2:-2, 2:-2]
    left1 =  raw[1:-3, 2:-2, 2:-2]
    right1 = raw[3:-1, 2:-2, 2:-2]
    right2 = raw[4:  , 2:-2, 2:-2]
    zin2 =   raw[2:-2, 2:-2,  :-4]
    zin1 =   raw[2:-2, 2:-2, 1:-3]
    zout2 =  raw[2:-2, 2:-2, 4:  ]
    zout1 =  raw[2:-2, 2:-2, 3:-1]

    for _ in xrange(i):
        tmp = (data+left1+right1+left2+right2+up2+up1+down2+down1)/9
        data[:] = tmp
    raw.cphvb = False
    
def main():

    benchmarks = [
        ('1. onethree', onethree, False), 
        ('2. onefive',  onefive, False),
        ('3. twofive',  twofive, False), 
        ('4. twonine',  twonine, True),
        ('5. threeseven',       threeseven, False),
        ('6. threethirtheen',   threethirtheen, False)
    ]

    b = util.Benchmark()
    n = b.size[0]
    m = b.size[1]
    i = b.size[2]

    b.start()
    twonine( n, m, i, b )
    b.stop()
    b.pprint()

if __name__ == "__main__":
    main()

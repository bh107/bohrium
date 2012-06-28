import numpy
import cphvbnumpy as cp
import time

cphvb = True

def onethree(itt):
    n=200**3
    raw = numpy.ones(n+2)
    raw.cphvb = cphvb
    data =  raw[1:-1]
    left =  raw[ :-2]
    right = raw[2:  ]

    for _ in xrange(itt):
        tmp = (data+left+right)/3
        data[:] = tmp
    print raw

def onefive(itt):
    n=200**3
    raw = numpy.ones(n+4)
    raw.cphvb = cphvb
    data =   raw[2:-2]
    left2 =  raw[ :-4]
    left1 =  raw[1:-3]
    right1 = raw[3:-1]
    right2 = raw[4:  ]

    for _ in xrange(itt):
        tmp = (data+left1+right1+left2+right2)/5
        data[:] = tmp
    print raw

def twofive(itt):
    n=4000
    raw = numpy.ones((n+2,n+2))
    raw.cphvb = cphvb
    data =  raw[1:-1, 1:-1]
    left =  raw[ :-2, 1:-1]
    right = raw[2:  , 1:-1]
    up =    raw[1:-1,  :-2]
    down =  raw[1:-1, 2:  ]

    for _ in xrange(itt):
        tmp = (data+left+right+up+down)/5
        data[:] = tmp
    print raw

def twonine(itt):
    n=4000
    raw = numpy.ones((n+4, n+4))
    raw.cphvb=cphvb

    data =   raw[2:-2, 2:-2]
    up2 =    raw[2:-2,  :-4]
    up1 =    raw[2:-2, 1:-3]
    down1 =  raw[2:-2, 3:-1]
    down2 =  raw[2:-2, 4:  ]
    left2 =  raw[ :-4, 2:-2]
    left1 =  raw[1:-3, 2:-2]
    right1 = raw[3:-1, 2:-2]
    right2 = raw[4:  , 2:-2]

    for _ in xrange(itt):
        tmp = (data+left1+right1+left2+right2+up2+up1+down2+down1)/9
        data[:] = tmp
    print raw

def threeseven(itt):
    n=200
    raw = numpy.ones((n+2,n+2, n+2))
    raw.cphvb=cphvb
    data =  raw[1:-1, 1:-1, 1:-1]
    left =  raw[ :-2, 1:-1, 1:-1]
    right = raw[2:  , 1:-1, 1:-1]
    up =    raw[1:-1,  :-2, 1:-1]
    down =  raw[1:-1, 2:  , 1:-1]
    zin =   raw[1:-1, 1:-1,  :-2]
    zout =  raw[1:-1, 1:-1, 2:  ]

    for _ in xrange(itt):
        tmp = (data+left+right+up+down+zin+zout)/7
        data[:] = tmp
    print raw

def threethirtheen(itt):
    n=200
    raw = numpy.ones((n+4, n+4, n+4))
    raw.cphvb=cphvb
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

    for _ in xrange(itt):
        tmp = (data+left1+right1+left2+right2+up2+up1+down2+down1)/9
        data[:] = tmp
    print raw

def benchmark(itt):

    benchmarks = [
        ('1. onethree', onethree), 
        ('2. onefive',  onefive),
        ('3. twofive',  twofive), 
        ('4. twonine',  twonine),
        ('5. threeseven',       threeseven),
        ('6. threethirtheen',   threethirtheen)
    ]

    for name, bench in benchmarks:
        start = time.time()
        bench(itt)
        stop = time.time()
        print name,': time taken',stop-start

benchmark(1)
    

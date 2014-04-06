import itertools

def cartesian(x, size):
    dist = []
    for i in xrange(size):
        start = i
        stop = -size+1+i
        if stop==0: stop = None
        dist.append((start,stop))
    stencil = [x[s] for s in [map((lambda se : slice(se[0],se[1])),i) 
               for i in itertools.product(dist,  
               repeat=len(x.shape))]]
    return stencil

def no_border(x, border):
    stencil = [x[s] for s in [map((lambda se : slice(se[0],se[1])),i) 
               for i in itertools.product([(border,-border)],  
               repeat=len(x.shape))]]
    return stencil[0]

    
def D2P9(x):
    if len(x.shape)!=2:
        raise Exception('Invalid shape for stencil'+str(len(x)))
    return cartesian(x,3)


def D3P27(x):
    if len(x.shape)!=3:
        print len(x)
        raise Exception('Invalid shape for stencil')
    return cartesian(x,3)

def D2P8(x):
    result = D2P9(x)
    result.pop(4)
    return result

def D3P26(x):
    result = D3P27(x)
    result.pop(13)
    return result

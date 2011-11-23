#!/usr/bin/env python
from sets import Set


def nelems(ndim, shape):

    elems = 1
    for d in xrange(0, ndim):
        elems *= shape[d]

    return elems

def coords( ndim, shape ):

    shape.append(0)

    c       = 0
    ce      = nelems(ndim, shape)
    coord   = [0]
    coord   = [0]*(ndim+1)

    next_dim = 1
    last_dim = ndim-1

    coord_set   = set([])
    coord_list  = []

    while c < ce:

        for ed in xrange(0, shape[0]):
            coord_list.append( tuple(coord[0:ndim]) )
            coord[0] += 1
            c += 1
        
        coord[0] = 0

        for d in xrange(1, ndim):

            coord[d] += 1
            if coord[d] < shape[d]:
                break
            else:
                coord[d] = 0

    return coord_list

def main():

    d       = [3]
    dd      = [3, 3]
    ddd     = [3, 3, 3]
    dddd    = [3, 3, 3, 3]
    ddddd   = [3, 3, 3, 3, 3]
    dddddd  = [3, 3, 3, 3, 3, 3]

    #print_coords( len(d), d )
    #print_coords( len(dd), dd )
    #print_coords( len(ddd), ddd )
    #print_coords( len(dddd), dddd )
    #print_coords( len(ddddd), ddddd )
    cc = coords( len(dddddd), dddddd )
    cc_set = set(cc)

    #print len(cc_set), cc_set

    for coord in cc:
            print coord

    if len(cc) == len(cc_set):
    
        print "Seems correct."

if __name__ == "__main__":
    main()

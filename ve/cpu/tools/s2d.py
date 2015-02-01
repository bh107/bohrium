#!/usr/bin/env python
import numpy
import pprint

def prod(values):
    valp = 1
    for value in values:
        valp *= value

    return valp

def rscan(shape):
    hej = [prod(shape[i:]) for i in xrange(0, len(shape))]+[1]
    return hej[1:]

def eidx_to_coord(eidx, ndim):
    coord = [(eidx / weight[dim]) % shape[dim] for dim in xrange(0, ndim)]
    return coord

nthreads = 32
shape = [18]
rank = len(shape)
nelements = prod(shape)
nrows = prod(shape[:-1])
spill = nelements % nthreads
weight = rscan(shape)

w = [1]*rank
acc = 1
for i in xrange(rank-1, -1, -1):
    w[i] = acc
    acc *= shape[i]

part = [{
    'work': 0,
    'begin': 0,
    'end': 0,
    'coord_begin': (0,0,0),
    'coord_end': (0,0,0)
}]*nthreads

def ceilDiv(a, b):
    return -(-a/b)

def partition(tid):

    work  = nelements / nthreads # Take what can be shared equal
    spill = nelements % nthreads

    begin = end = 0
    if tid < spill:
        work += 1
        begin = tid * work
        end = begin + work -1
    elif work>0:
        begin = tid * work + spill
        end = begin + work -1
    else:
        print "No work!"

    coord_begin = eidx_to_coord(begin, rank)
    coord_end = eidx_to_coord(end, rank)

    rows = ceilDiv(work, shape[-1])

    return {
        'work': work,
        'begin': begin,
        'end': end,
        'coord_begin': coord_begin,
        'coord_end': coord_end,
        'rows_accessed': rows
    }

for tid in xrange(0, nthreads):
    part[tid] = partition(tid)

pprint.pprint(part)

print "nthreads =", nthreads
print "rank =", rank
print "nelements =", nelements
print "shape =", shape
print "nrows =", nrows
print "spill =", spill
print "weight =", weight, w


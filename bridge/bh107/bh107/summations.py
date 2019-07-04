"""
Summations and products
~~~~~~~~~~~~~~~~~~~~~~~
"""
import bh107

def sum(a, axis=None, dtype=None, out=None):
    return bh107.add.reduce(a, axis=axis, out=out)

def prod(a, axis=None, dtype=None, out=None):
    return bh107.multiply.reduce(a, axis=axis, out=out)

def max(a, axis=None, out=None):
    return bh107.maximum.reduce(a, axis=axis, out=out)

def min(a, axis=None, out=None):
    return bh107.minimum.reduce(a, axis=axis, out=out)

def any(a, axis=None, out=None, keepdims=None):
    return bh107.logical_or.reduce(a.astype(bool), axis=axis, out=out)

def all(a, axis=None, out=None, keepdims=None):
    return bh107.logical_and.reduce(a.astype(bool), axis=axis, out=out)

#TODO: Merge from bohrium/summations.py
#
# def argmax(a, axis=None, out=None):
# def argmin(a, axis=None, out=None):
# def mean(a, axis=None, dtype=None, out=None):
# average = mean

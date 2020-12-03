import time
import numpy as np
import bohrium as bh
import example as bohr

ITER = 10000
SHAPE = (10000,)


def compute(m, a, b):
    return m.add(m.add(m.add(a, b), 42), b)


def numpy_bench():
    a = np.full(SHAPE, 42)
    b = np.full(SHAPE, 43)
    for _ in range(ITER):
        compute(np, a, b)


def bohr_bench():
    a = bohr.full(SHAPE, 42)
    b = bohr.full(SHAPE, 43)
    for _ in range(ITER):
        compute(bohr, a, b)
        bohr.flush()


def bh_bench():
    a = bh.full(SHAPE, 42)
    b = bh.full(SHAPE, 43)
    for _ in range(ITER):
        compute(bh, a, b)
        bh.flush()


t1 = time.time()
numpy_bench()
t2 = time.time()
print("Numpy:   ", t2 - t1)

t1 = time.time()
bohr_bench()
t2 = time.time()
print("Bohr:    ", t2 - t1)

t1 = time.time()
bh_bench()
t2 = time.time()
print("Bohrium: ", t2 - t1)

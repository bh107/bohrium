import bohrium as np
from sys import stderr
from . import ufuncs


def __lapack(name, a, b):
    if not b.flags['C_CONTIGUOUS']:
        b = b.copy(order='C')

    if not a.flags['C_CONTIGUOUS']:
        a = a.copy(order='C')

    ufuncs.extmethod(name, b, a, b)  # modifies 'b' unless copy is True
    return b


def gesv(a, b):
    return __lapack("lapack_gesv", a.T.copy(order='C'), b)


def gbsv(a, b):
    return __lapack("lapack_gbsv", a, b)


def gtsv(a, b):
    return __lapack("lapack_gtsv", a, b)


def posv(a, b):
    return __lapack("lapack_posv", a, b)


def ppsv(a, b):
    return __lapack("lapack_ppsv", a, b)


def spsv(a, b):
    return __lapack("lapack_spsv", a, b)

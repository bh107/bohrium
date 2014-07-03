"""
Random
~~~~~~

Random functions

"""
import bohrium as np
import numpy
import operator
import datetime
import os
import backend

from libc.stdint cimport uint64_t, uint32_t

cdef extern from "Random123/philox.h":
    struct r123array2x32:
        pass
    struct r123array1x32:
        pass
ctypedef r123array2x32 philox2x32_ctr_t
ctypedef r123array1x32 philox2x32_key_t
cdef extern from "Random123/philox.h":
    philox2x32_ctr_t philox2x32(philox2x32_ctr_t, philox2x32_key_t) 
cdef union ctr_t:
    philox2x32_ctr_t c
    uint64_t ul 
cdef union key_t:
    philox2x32_key_t k
    uint32_t ui 

cdef extern from "Python.h": 
    void* PyLong_AsVoidPtr(object)

def random123(shape, uint64_t start_index, uint32_t key, bohrium=True):
    """
    New array of uniform pseudo numbers based on the random123 philox2x32 algorithm.
    NB: dtype is np.uint64 always

    Parameters
    ----------
    shape       : tuple of ints
                  Defines the shape of the returned array of random floats.
    start_index : The start index (must be positive)
    key         : The key or seed for the random123 algorithm

    Returns
    -------
    out : Array of uniform pseudo numbers
    """
    cdef ctr_t ctr 
    cdef ctr_t rnd
    cdef key_t k
    cdef uint64_t* array_data
    cdef long length
    cdef long i

    ctr.ul = start_index
    k.ui = key

    if shape is None:
        rnd.c = philox2x32(ctr.c,k.k)
        return rnd.ul
    elif bohrium is False:
        array = np.empty(shape,dtype=np.uint64,bohrium=False)
        length = array.size
        array_data = <uint64_t *> PyLong_AsVoidPtr(array.ctypes.data)
        for i from 0 <= i < length:
            rnd.c = philox2x32(ctr.c,k.k)
            array_data[i] = rnd.ul
            ctr.ul += 1
        return array
    else:
        totalsize = numpy.multiply.reduce(numpy.asarray(shape))
        bhc_obj = backend.random123(totalsize, start_index, key)
        ret = np.ndarray.new((totalsize,), np.uint64, bhc_obj)
        return ret

class Random:
    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, x=None):
        """
        Initialize the random number generator object. Optional argument x
        can be any hashable object. If x is omitted or None, current
        system time is used; current system time is also used to initialize
        the generator when the module is first imported. If randomness
        sources are provided by the operating system, they are used instead
        of the system time (see the os.urandom() function for details on
        availability).
        """
        if x is None:
            try:
                self.key = numpy.uint32(hash(os.urandom(8)))
            except NotImplementedError:
                self.key = numpy.uint32(hash(datetime.datetime.now()))
        else:
            self.key = numpy.uint32(x)
        self.index = 0;

    def random_sample(self, shape=None, dtype=np.float64, bohrium=True):
        """
        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random` by `(b-a)` and add `a`::

          (b - a) * random() + a

        Parameters
        ----------
        shape : int or tuple of ints, optional
            Defines the shape of the returned array of random floats. If None
            (the default), returns a single float.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `shape` (unless ``shape=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> np.random.random()
        0.47108547995356098
        >>> type(np.random.random())
        <type 'float'>
        >>> np.random.random((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        """

        #Generate random numbers as uint
        r_int = random123(shape, self.index, self.key, bohrium=bohrium)
        #Convert random numbers to float in the interval [0.0, 1.0).
        max_value = numpy.dtype(dtype).type(numpy.iinfo(numpy.uint64).max)
        if shape is None:
            self.index += 1
            return numpy.dtype(dtype).type(r_int) / max_value 
        else:
            self.index += numpy.multiply.reduce(numpy.asarray(shape))
            r = np.empty_like(r_int, dtype=dtype)
            r[...] = r_int
            r /= max_value
            return r

#The default random object
_inst = Random()
seed = _inst.seed
random_sample = _inst.random_sample
ranf = random = sample = random_sample

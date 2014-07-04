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
import sys
import backend
import math

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

cdef class RandomState:
    """
    RandomState(seed=None)

    Container for the Random123 pseudo-random number generator.

    `RandomState` exposes a number of methods for generating random numbers
    drawn from a variety of probability distributions. In addition to the
    distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned. If `size` is a tuple,
    then an array with that shape is filled and returned.

    Parameters
    ----------
    seed : int, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer or ``None`` (the default).
        If `seed` is ``None``, then `RandomState` will try to read data from
        ``/dev/urandom`` (or the Windows analogue) if available or seed from
        the clock otherwise.

    """
    cdef uint32_t key
    cdef uint64_t index
    cdef bint has_gauss
    cdef double gauss

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.

        Parameters
        ----------
        seed : int or array_like, optional
            Seed for `RandomState`.

        See Also
        --------
        RandomState

        """
        if seed is None:
            try:
                self.key = numpy.uint32(hash(os.urandom(8)))
            except NotImplementedError:
                self.key = numpy.uint32(hash(datetime.datetime.now()))
        else:
            self.key = numpy.uint32(seed)
        self.index = 0;
        self.has_gauss = False

    def get_state(self):
        """
        get_state()

        Return a tuple representing the internal state of the generator.

        For more details, see `set_state`.

        Returns
        -------
        out : tuple(str, np.uint64, np.uint32)
            The returned tuple has the following items:

            1. the string 'Random123'.
            2. an integer ``index``.
            3. an integer ``key``.


        See Also
        --------
        set_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in Bohrium. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        """
        return ('Random123', self.index, self.key)

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of the
        "Mersenne Twister"[1]_ pseudo-random number generating algorithm.

        Parameters
        ----------
        state : tuple(str, np.uint64, np.uint32)
            The returned tuple has the following items:

            1. the string 'Random123'.
            2. an integer ``index``.
            3. an integer ``key``.

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in Bohrium. If the internal state is manually altered,
        the user should know exactly what he/she is doing.
        """
        if len(state) != 3:
            raise ValueError("state must contain 3 elements")
        if state[0] != 'Random123':
            raise ValueError("algorithm must be 'Random123'")
        try:
            self.index = state[1]
            self.key = state[2]
        except TypeError:
            raise ValueError("state is not a valid Random123 state")
            

    def random123(self, size=None, bohrium=True):
        """
        New array of uniform pseudo numbers based on the random123 philox2x32 algorithm.
        NB: dtype is np.uint64 always
        
        Parameters
        ----------
        shape       : tuple of ints
        Defines the shape of the returned array of random floats.
        
        Returns
        -------
        out : Array of uniform pseudo numbers
        """
        cdef ctr_t ctr 
        cdef ctr_t rnd
        cdef key_t key
        cdef uint64_t* array_data
        cdef long length
        cdef long i
        
        ctr.ul = self.index
        key.ui = self.key
        
        if size is None:
            length = 1
            rnd.c = philox2x32(ctr.c,key.k)
            ret = rnd.ul
        elif bohrium is False:
            ret = np.empty(size,dtype=np.uint64,bohrium=False)
            length = ret.size
            array_data = <uint64_t *> PyLong_AsVoidPtr(ret.ctypes.data)
            for i from 0 <= i < length:
                rnd.c = philox2x32(ctr.c,key.k)
                array_data[i] = rnd.ul
                ctr.ul += 1
        else:
            length = numpy.multiply.reduce(numpy.asarray(size))
            bhc_obj = backend.random123(length, self.index, self.key)
            ret = np.ndarray.new((length,), np.uint64, bhc_obj).reshape(size)
        self.index += length
        return ret

    def random_sample(self, size=None, dtype=float, bohrium=True):
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
        dtype=np.dtype(dtype).type
        if not (dtype is np.float64 or dtype is np.float32):
            raise ValueError("dtype not supported for random_sample")
        #Generate random numbers as uint
        r_uint = self.random123(size, bohrium=bohrium)
        #Convert random numbers to float in the interval [0.0, 1.0).
        max_value = dtype(numpy.iinfo(numpy.uint64).max)
        if size is None:
            return numpy.dtype(dtype).type(r_uint) / max_value 
        else:
            r = np.empty_like(r_uint, dtype=dtype)
            r[...] = r_uint
            r /= max_value
            return r

    def tomaxint(self, size=None, bohrium=True):
        """
        tomaxint(size=None, bohrium=True)

        Random integers between 0 and ``sys.maxint``, inclusive.

        Return a sample of uniformly distributed random integers in the interval
        [0, ``sys.maxint``].

        Parameters
        ----------
        size : tuple of ints, int, optional
            Shape of output.  If this is, for example, (m,n,k), m*n*k samples
            are generated.  If no shape is specified, a single sample is
            returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        Examples
        --------
        >>> RS = np.random.mtrand.RandomState() # need a RandomState object
        >>> RS.tomaxint((2,2,2))
        array([[[1170048599, 1600360186],
                [ 739731006, 1947757578]],
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> import sys
        >>> sys.maxint
        2147483647
        >>> RS.tomaxint((2,2,2)) < sys.maxint
        array([[[ True,  True],
                [ True,  True]],
               [[ True,  True],
                [ True,  True]]], dtype=bool)

        """
        r_uint = self.random123(size,bohrium) >> (64 - sys.maxint.bit_length())
        res = np.empty_like(r_uint, dtype=int)
        res[...] = r_uint
        return res 

    def randint(self, low, high=None, size=None, dtype=int, bohrium=True):
        """
        randint(low, high=None, size=None, dtype=int, bohrium=True)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted. In particular, this other one is the one to use to generate
            uniformly distributed discrete non-integers.

        Examples
        --------
        >>> np.random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> np.random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> np.random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])

        """
        dtype = np.dtype(dtype).type
        if high is None:
            high = low
            low = 0
        if low >= high :
            raise ValueError("low >= high")
        diff = high - low
        if size is None:
            return dtype(self.random123(size,bohrium=bohrium) % diff) + low
        else:
            return np.array(self.random123(size,bohrium=bohrium) % diff, dtype=dtype, bohrium=bohrium) + low

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float, bohrium=True):
        """
        uniform(low=0.0, high=1.0, size=None, dtype=float, bohrium=True)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be
            less than high.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,
               uniformly distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        same as:
        random_sample(size) * (high - low) + low

        Examples
        --------
        Draw samples from the distribution:

        >>> s = np.random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, normed=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()

        """
        dtype = np.dtype(dtype).type
        return self.random_sample(size=size, dtype=dtype, bohrium=bohrium) * dtype(high - low) + dtype(low)

    def rand(self, *args, dtype=float, bohrium=True):
        """
        rand(d0, d1, ..., dn, dtype=float, bohrium=True)

        Random values in a given shape.

        Create an array of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Notes
        -----
        This is a convenience function. If you want an interface that
        takes a shape-tuple as the first argument, refer to
        np.random.random_sample .

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random

        """
        if len(args) == 0:
            return self.random_sample(dtype=dtype, bohrium=bohrium)
        else:
            return self.random_sample(size=args, dtype=dtype, bohrium=bohrium)

    def randn(self, *args, dtype=float, bohrium=True):
        """
        randn(d0, d1, ..., dn, dtype=float, bohrium=True)

        Return a sample (or samples) from the "standard normal" distribution.

        If positive, int_like or int-convertible arguments are provided,
        `randn` generates an array of shape ``(d0, d1, ..., dn)``, filled
        with random floats sampled from a univariate "normal" (Gaussian)
        distribution of mean 0 and variance 1 (if any of the :math:`d_i` are
        floats, they are first converted to integers by truncation). A single
        float randomly sampled from the distribution is returned if no
        argument is provided.

        This is a convenience function.  If you want an interface that takes a
        tuple as the first argument, use `numpy.random.standard_normal` instead.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should be all positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        Z : ndarray or float
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
            the standard normal distribution, or a single such float if
            no parameters were supplied.

        See Also
        --------
        random.standard_normal : Similar, but takes a tuple as its argument.

        Notes
        -----
        For random samples from :math:`N(\\mu, \\sigma^2)`, use:

        ``sigma * np.random.randn(...) + mu``

        Examples
        --------
        >>> np.random.randn()
        2.1923875335537315 #random

        Two-by-four array of samples from N(3, 6.25):

        >>> 2.5 * np.random.randn(2, 4) + 3
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random

        """
        if len(args) == 0:
            return self.standard_normal(dtype=dtype, bohrium=bohrium)
        else:
            return self.standard_normal(size=args, dtype=dtype, bohrium=bohrium)

    def random_integers(self, low, high=None, size=None, dtype=int, bohrium=True):
        """
        random_integers(low, high=None, size=None, dtype=int, bohrium=True)

        Return random integers between `low` and `high`, inclusive.

        Return random integers from the "discrete uniform" distribution in the
        closed interval [`low`, `high`].  If `high` is None (the default),
        then results are from [1, `low`].

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, the largest (signed) integer to be drawn from the
            distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.randint : Similar to `random_integers`, only for the half-open
            interval [`low`, `high`), and 0 is the lowest value if `high` is
            omitted.

        Notes
        -----
        To sample from N evenly spaced floating-point numbers between a and b,
        use::

          a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

        Examples
        --------
        >>> np.random.random_integers(5)
        4
        >>> type(np.random.random_integers(5))
        <type 'int'>
        >>> np.random.random_integers(5, size=(3.,2.))
        array([[5, 4],
               [3, 3],
               [4, 5]])

        Choose five random numbers from the set of five evenly-spaced
        numbers between 0 and 2.5, inclusive (*i.e.*, from the set
        :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

        >>> 2.5 * (np.random.random_integers(5, size=(5,)) - 1) / 4.
        array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ])

        Roll two six sided dice 1000 times and sum the results:

        >>> d1 = np.random.random_integers(1, 6, 1000)
        >>> d2 = np.random.random_integers(1, 6, 1000)
        >>> dsums = d1 + d2

        Display results as a histogram:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(dsums, 11, normed=True)
        >>> plt.show()

        """
        if high is None:
            high = low
            low = 1
        return self.randint(low, high+1, size)

    # Complicated, continuous distributions:
    def standard_normal(self, size=None, dtype=float, bohrium=True):
        """
        standard_normal(size=None, dtype=float, bohrium=True)

        Returns samples from a Standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        dtype = np.dtype(dtype).type
        if not (dtype is np.float64 or dtype is np.float32):
            raise ValueError("dtype not supported for standart_normal")
        if size is None:
            if self.has_gauss:
                self.has_gauss = False
                return dtype(self.gauss)
            else:
                u1 = self.random_sample()
                u2 = self.random_sample()
                r = math.sqrt(-2. * math.log(u1))
                t = 2. * math.pi * u2
                z0 = r * math.cos(t)
                z1 = r * math.sin(t)
                self.gauss = z1
                self.has_gauss = True
                return dtype(z0)
        else:
            length = numpy.multiply.reduce(numpy.asarray(size))
            hlength = length / 2 + length % 2
            u1 = self.random_sample(size=hlength, dtype=dtype, bohrium=bohrium)
            u2 = self.random_sample(size=hlength, dtype=dtype, bohrium=bohrium)
            r = np.sqrt(-2. * np.log(u1))
            t = 2. * math.pi * u2
            z0 = r * np.cos(t)
            z1 = r * np.sin(t)
            res = np.empty(hlength*2, dtype=dtype, bohrium=bohrium)
            res[:hlength] = z0 # res[::2] = z0
            res[hlength:] = z1 # res[1::2] = z1
            return res[:length].reshape(size)



#The default random object
_inst = RandomState()
seed = _inst.seed
get_state = _inst.get_state
set_state = _inst.set_state
random_sample = _inst.random_sample
ranf = random = sample = random_sample
randint = _inst.randint
uniform = _inst.uniform
rand = _inst.rand
randn = _inst.randn
random_integers = _inst.random_integers
standard_normal = _inst.standard_normal

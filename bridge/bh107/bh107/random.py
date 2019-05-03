# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
from . import bharray, _dtype_util, util
from .ufuncs import ufunc_dict
from bohrium_api import _bh_api


class RandomState:
    def __init__(self, seed=None):
        """Container for the Random123 pseudo-random number generator.

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

        self.key = None
        self.index = None
        self.has_gauss = None
        self.seed(seed)

    def seed(self, seed=None):
        """Seed the generator.

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
                self.key = np.uint32(hash(os.urandom(8)))
            except NotImplementedError:
                self.key = np.uint32(hash(datetime.datetime.now()))
        else:
            self.key = np.uint32(seed)
        self.index = 0
        self.has_gauss = False

    def get_state(self):
        """Return a tuple representing the internal state of the generator.

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
        """Set the internal state of the generator from a tuple.

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

    def random123(self, shape):
        """New array of uniform pseudo numbers based on the random123 philox2x32 algorithm.
        NB: dtype is np.uint64 always

        Parameters
        ----------
        shape       : int or tuple of ints
        Defines the shape of the returned array of random floats.

        Returns
        -------
        out : Array of uniform pseudo numbers
        """
        if np.isscalar(shape):
            shape = (shape,)

        length = util.total_size(shape)
        flat = bharray.BhArray(length, np.uint64)
        _bh_api.random123(flat._bhc_handle, self.index, self.key)
        self.index += flat.nelem
        return flat.reshape(shape)

    def random_sample(self, shape):
        """Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random() + a

        Parameters
        ----------
        shape : int or tuple of ints
            Defines the shape of the returned array of random floats.

        Returns
        -------
        out : BhArray of floats
            Array of random floats of shape `shape`.

        Examples
        --------
        >>> np.random.random((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        """
        # Generate random numbers as uint
        r_uint = self.random123(shape)
        # Convert random numbers to float in the interval [0.0, 1.0) and return.
        return r_uint.astype(np.float64) / np.iinfo(np.uint64).max

    def randint(self, low, high=None, shape=None):
        """Return random integers from `low` (inclusive) to `high` (exclusive).

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
        shape : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : BhArray of ints
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
        if high is None:
            high = low
            low = 0
        if low >= high:
            raise ValueError("low >= high")
        diff = high - low
        return self.random123(shape) % diff + low

    def uniform(self, low=0.0, high=1.0, shape=None):
        """Draw samples from a uniform distribution.

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
        shape : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : BhArray
            Drawn samples, with shape `shape`.

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
        return self.random_sample(shape).astype(np.float64) * (high - low) + low

    def rand(self, *shape):
        """Random values in a given shape.

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
        out : BhArray, shape ``(d0, d1, ..., dn)``
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
        return self.random_sample(shape)

    def random_integers(self, low, high=None, shape=None):
        """Return random integers between `low` and `high`, inclusive.

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
        shape : tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : BhArray of ints
            `size`-shaped array of random integers from the appropriate
            distribution.

        See Also
        --------
        random.randint : Similar to `random_integers`, only for the half-open
            interval [`low`, `high`), and 0 is the lowest value if `high` is
            omitted.

        Notes
        -----
        To sample from N evenly spaced floating-point numbers between a and b,
        use::

          a + (b - a) * (bh107.random.random_integers(N) - 1) / (N - 1.)

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
        return self.randint(low, high + 1, shape)

    def standard_exponential(self, shape=None):
        """ Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        shape : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : BhArray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = np.random.standard_exponential((3, 8000))

        """
        # We use -log(1-U) since U is [0, 1) */
        return -1 * ufunc_dict['log'](1 - self.random_sample(shape))

    def exponential(self, scale=1.0, shape=None):
        """ Exponential distribution.

        Its probability density function is

        .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
        The rate parameter is an alternative, widely used parameterization
        of the exponential distribution [3]_.

        The exponential distribution is a continuous analogue of the
        geometric distribution.  It describes many common situations, such as
        the size of raindrops measured over many rainstorms [1]_, or the time
        between page requests to Wikipedia [2]_.

        Parameters
        ----------
        scale : float
            The scale parameter, :math:`\\beta = 1/\\lambda`.
        shape : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        Returns
        -------
        out : BhArray
            Drawn samples.

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] "Poisson Process", Wikipedia,
               http://en.wikipedia.org/wiki/Poisson_process
        .. [3] "Exponential Distribution, Wikipedia,
               http://en.wikipedia.org/wiki/Exponential_distribution

        """

        if scale <= 0:
            raise ValueError("The `scale` must be greater than zero")
        return self.standard_exponential(shape) * scale

    def random(self, shape=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Alias for `random_sample`
        """
        return self.random_sample(shape)

    def sample(self, shape=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Alias for `random_sample`
        """
        return self.random_sample(shape)

    def ranf(self, shape=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Alias for `random_sample`
        """
        return self.random_sample(shape)

    def random_of_dtype(self, dtype, shape=None):
        """Return random array of `dtype`. The values are in the interval of the `dtype`.

        Parameters
        ----------
        dtype : data-type
            The desired data-type for the array.

        shape : int or tuple of ints
            Defines the shape of the returned array of random floats.

        Returns
        -------
        out : BhArray of floats
            Array of random floats of shape `shape`.
        """

        dtype = _dtype_util.obj_to_dtype(dtype)
        if dtype is np.bool:
            res = self.random_integers(0, 1, shape)
        elif dtype in [np.int8, np.uint8]:
            res = self.random_integers(1, 3, shape)
        elif dtype is np.int16:
            res = self.random_integers(1, 5, shape)
        elif dtype is np.uint16:
            res = self.random_integers(1, 6, shape)
        elif dtype in [np.float32, np.float64]:
            res = self.random_sample(shape)
        elif dtype in [np.complex64, np.complex128]:
            res = self.random_sample(shape=shape) + self.random_sample(shape=shape) * 1j
        else:
            res = self.random_integers(1, 8, shape)
        if len(res.shape) == 0:  # Make sure scalars is arrays.
            res = bharray.BhArray.from_object(res)
            res.shape = shape
        return res.astype(dtype)


# The default random object
_inst = RandomState()
seed = _inst.seed
get_state = _inst.get_state
set_state = _inst.set_state
random_sample = _inst.random_sample
random = _inst.random
sample = _inst.sample
ranf = _inst.ranf
randint = _inst.randint
uniform = _inst.uniform
rand = _inst.rand
random_integers = _inst.random_integers
standard_exponential = _inst.standard_exponential
exponential = _inst.exponential
random_of_dtype = _inst.random_of_dtype

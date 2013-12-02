"""
Random
~~~~~~

Random functions

"""
import bohrium as np
import operator
import datetime
import os
import bohriumbridge as bridge


def random123(shape, key, start_idx=0, bohrium=True):
    """
    New array of uniform pseudo numbers based on the random123 algorithm.

    Parameters
    ----------
    shape     : tuple of ints
                Defines the shape of the returned array of random floats.
    key       : The key or seed for the random123 algorithm
    start_idx : The start index (must be positive)

    Returns
    -------
    out : Array of uniform pseudo numbers
    """
    assert start_idx >= 0
    totalsize = reduce(operator.mul, shape, 1)
    out = np.empty(totalsize, bohrium=bohrium, dtype=np.uint64)
    arange = start_idx + np.arange(totalsize, bohrium=bohrium, dtype=np.uint64)
    bridge.random123(out, key, arange)
    return out.reshape(shape)



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
                self.seed = hash(os.urandom(8))
            except NotImplementedError:
                self.seed = hash(datetime.datetime.now())
        else:
            self.seed = hash(x)
        self.idx = 0;

    def random(self, shape=None, dtype=np.float64, bohrium=True):
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
        assert dtype is np.float32 or dtype is np.float64
        if shape is None:
            s = (1,) #default is a scalar
        else:
            try:
                s = (int(shape),) #Convert integer to tuple
            except TypeError:
                s = shape #It might be a tuble already

        r = random123(s, self.seed, start_idx=self.idx, bohrium=bohrium)
        r = np.asarray(r, dtype=dtype)
        r /= float(np.iinfo(np.uint64).max)

        self.idx += reduce(operator.mul, s, 1)

        if shape is None:
            return r[0]
        else:
            return r

#The default random object
_inst = Random()
seed = _inst.seed
random = _inst.random

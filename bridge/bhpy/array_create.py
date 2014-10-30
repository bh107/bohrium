"""
Array Create Routines
~~~~

"""
import math
from . import ndarray
from .ndarray import fix_returned_biclass
import numpy
from ._util import dtype_equal, dtype_in
from . import backend

@fix_returned_biclass
def array(object, dtype=None, copy=False, order=None, subok=False, ndmin=0, bohrium=True):
    """
    Create an array -- Bohrium or NumPy ndarray.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true, then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`dtype`, `order`, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.
    bohrium : boolean, optional
        Determines whether it is a Bohrium array (bohrium.ndarray) or a
        regular NumPy array (numpy.ndarray)

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    See Also
    --------
    empty, empty_like, zeros, zeros_like, ones, ones_like, fill

    Examples
    --------
    >>> np.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> np.array([1, 2, 3.0])
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> np.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> np.array([1, 2, 3], ndmin=2)
    array([[1, 2, 3]])

    Type provided:

    >>> np.array([1, 2, 3], dtype=complex)
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    Data-type consisting of more than one element:

    >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
    >>> x['a']
    array([1, 3])

    Creating an array from sub-classes:

    >>> np.array(np.mat('1 2; 3 4'))
    array([[1, 2],
           [3, 4]])

    >>> np.array(np.mat('1 2; 3 4'), subok=True)
    matrix([[1, 2],
            [3, 4]])

    """
    a = object
    if bohrium:
        if ndarray.check(a):
            if order == 'F':
                raise ValueError("Cannot convert a Bohrium array to "\
                            "column-major ('F') memory representation")
            if copy:
                a = a.copy()
            if dtype is not None and not dtype_equal(dtype, a.dtype):
                t = empty_like(a,dtype=dtype)
                t[...] = a
                a = t
            for i in xrange(a.ndim, ndmin):
                a = numpy.expand_dims(a,i)
            return a
        else:
            a = numpy.array(a, dtype=dtype, copy=copy, order=order,\
                            subok=subok, ndmin=ndmin)
            a = numpy.require(a, requirements=['C_CONTIGUOUS', 'ALIGNED', \
                                                'OWNDATA'])
            ret = empty(a.shape, dtype=a.dtype)
            ret._data_fill(a)
            return ret
    else:
        if ndarray.check(a):
            ret = a.copy2numpy()
            return numpy.array(ret, dtype=dtype, copy=copy, order=order, \
                               subok=subok, ndmin=ndmin)
        else:
            return numpy.array(a, dtype=dtype, copy=copy, order=order, \
                               subok=subok, ndmin=ndmin)

@fix_returned_biclass
def empty(shape, dtype=float, bohrium=True):
    """
    Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.

    See Also
    --------
    empty_like, zeros

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    `empty`, unlike `zeros`, does not set the matrix values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307],
            [  7.39337286e-309,   3.22135945e-309]])        #random
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0],
            [ 6586976, 22740995]])                          #random

    """
    if not bohrium:
        return numpy.empty(shape, dtype=dtype)
    return ndarray.new(shape, dtype)

@fix_returned_biclass
def ones(shape, dtype=float, bohrium=True):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.
    matlib.zeros : Zero matrix.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.ones((2,3))
    matrix([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> np.matlib.ones(2)
    matrix([[ 1.,  1.]])

    """
    A = empty(shape, dtype=dtype, bohrium=bohrium)
    A[...] = A.dtype.type(1)
    return A

@fix_returned_biclass
def zeros(shape, dtype=float, bohrium=True):
    """
    Return a matrix of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is float.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : matrix
        Zero matrix of given shape, dtype, and order.

    See Also
    --------
    numpy.zeros : Equivalent array function.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.zeros((2, 3))
    matrix([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> np.matlib.zeros(2)
    matrix([[ 0.,  0.]])

    """
    A = empty(shape, dtype=dtype, bohrium=bohrium)
    A[...] = A.dtype.type(0)
    return A

@fix_returned_biclass
def empty_like(a, dtype=None, bohrium=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    #random
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    """
    if dtype is None:
        dtype = a.dtype
    if bohrium is None:
        bohrium = ndarray.check(a)
    return empty(a.shape, dtype, bohrium)

@fix_returned_biclass
def zeros_like(a, dtype=None, bohrium=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    With default parameters, is equivalent to ``a.copy().fill(0)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> np.zeros_like(y)
    array([ 0.,  0.,  0.])

    """
    if dtype is None:
        dtype = a.dtype
    if bohrium is None:
        bohrium = ndarray.check(a)
    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[...] = b.dtype.type(0)
    return b

@fix_returned_biclass
def ones_like(a, dtype=None, bohrium=None):
    """
    Return an array of ones with the same shape and type as a given array.

    With default parameters, is equivalent to ``a.copy().fill(1)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> np.ones_like(y)
    array([ 1.,  1.,  1.])

    """
    if dtype is None:
        dtype = a.dtype
    if bohrium is None:
        bohrium = ndarray.check(a)
    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[...] = b.dtype.type(1)
    return b

@fix_returned_biclass
def arange(start, stop=None, step=1, dtype=None, bohrium=True):
    """
    arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    out : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Arrays of evenly spaced numbers in N-dimensions
    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions

    Examples
    --------
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])

    """
    if not bohrium:
        return numpy.arange(start,stop,step,dtype)
    if not stop:
        stop = start
        start = type(stop)(0)
    size = int(math.ceil((float(stop) - float(start)) / float(step)))
    if dtype is None:
        dtype = numpy.int64
    else:
        start = numpy.dtype(dtype).type(start)
        stop  = numpy.dtype(dtype).type(stop)
        step  = numpy.dtype(dtype).type(step)
    return range(size,dtype=dtype) * step + start

@fix_returned_biclass
def range(size, dtype=numpy.uint64):
    if (not isinstance(size, (int,long))):
        raise ValueError("size must be an integer")
    if (size < 1):
        raise ValueError("size must be greater than 0")
    if dtype_in(dtype, [numpy.int8,
                        numpy.int16,
                        numpy.int32,
                        numpy.uint8,
                        numpy.uint16,
                        numpy.uint32,
                        numpy.float16,
                        numpy.float32,
                        numpy.complex64]):
        A = empty(size,dtype=numpy.uint32,bohrium=True)
    else:
        A = empty(size,dtype=numpy.uint64,bohrium=True)
    ret = backend.range(size, A.dtype)
    A = ndarray.new((size,), A.dtype, ret)
    if not dtype_equal(dtype, A.dtype):
        B = empty_like(A,dtype=dtype)
        B[...] = A[...]
        return B
    else:
        return A

@fix_returned_biclass
def linspace(start, stop, num=50, endpoint=True, retstep=False,\
             dtype=float, bohrium=True):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop` ].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float (only if `retstep` is True)
        Size of spacing between samples.


    See Also
    --------
    arange : Similiar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
        array([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    if not bohrium:
        #TODO: add copy=False to .astype()
        return numpy.linspace(start, stop, num=num, endpoint=endpoint, \
                              retstep=retstep).astype(dtype)
    num = int(num)
    if num <= 0:
        return array([], dtype=dtype)
    if endpoint:
        if num == 1:
            return array([numpy.dtype(dtype).type(start)])
        step = (stop-start)/float((num-1))
    else:
        step = (stop-start)/float(num)
    y = arange(start, stop, step, dtype=dtype)
    if retstep:
        return y, step
    else:
        return y

def load(file, mmap_mode=None, bohrium=True):
    """
    Load an array(s) or pickled objects from .npy, .npz, or pickled files.

    Parameters
    ----------
    file : file-like object or string
        The file to read. Compressed files with the filename extension
        ``.gz`` are acceptable. File-like objects must support the
        ``seek()`` and ``read()`` methods. Pickled files require that the
        file-like object support the ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode (see
        `numpy.memmap` for a detailed description of the modes).  A
        memory-mapped array is kept on disk. However, it can be accessed
        and sliced like any ndarray.  Memory mapping is especially useful
        for accessing small fragments of large files without reading the
        entire file into memory.

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.

    See Also
    --------
    save, savez, savez_compressed, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.

    Notes
    -----
    - If the file contains pickle data, then whatever object is stored
      in the pickle is returned.
    - If the file is a ``.npy`` file, then a single array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.
    - If the file is a ``.npz`` file, the returned value supports the
      context manager protocol in a similar fashion to the open function::

        with load('foo.npz') as data:
            a = data['a']

      The underlying file descriptor is closed when exiting the 'with'
      block.

    Examples
    --------
    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Store compressed data to disk, and load it again:

    >>> a=np.array([[1, 2, 3], [4, 5, 6]])
    >>> b=np.array([1, 2])
    >>> np.savez('/tmp/123.npz', a=a, b=b)
    >>> data = np.load('/tmp/123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])

    """
    arr = numpy.load(file,mmap_mode)
    if bohrium:
        arr = array(arr, bohrium=True)
    return arr

def save(file, arr):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file or str
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.  If file is a string, a ``.npy``
        extension will be appended to the file name if it does not already
        have one.
    arr : array_like
        Array data to be saved.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see `format`.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    bohrium = ndarray.check(arr)
    if bohrium:
        array(arr, bohrium=False)
    numpy.save(file,arr)
    if bohrium:
        array(arr, bohrium=True)

###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest
from . import _info

class Tests(unittest.TestCase):

    def test_empty_dtypes(self):
        for t in _info.numpy_types:
            a = empty((4,4), dtype=t)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)

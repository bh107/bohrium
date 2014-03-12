"""
Core
~~~~

The ``core`` module provide the essential functions, such as all the array creation functions, diagonal and matrix multiplication.

"""
import numpy
from numpy import *
import bohriumbridge as bridge
from math import ceil
def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, bohrium=True):
    """
    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object
        whose __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array. If not given, then the type
        will be determined as the minimum type required to hold the objects
        in the sequence. This argument can only be used to 'upcast' the array.
        For downcasting, use the .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied. Otherwise, a copy will only
        be made if __array__ returns a copy, if obj is a nested sequence, or if a
        copy is needed to satisfy any of the other requirements (dtype, order, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array. If order is 'C' (default), then the array
        will be in C-contiguous order (last-index varies the fastest). If order is 'F',
        then the returned array will be in Fortran-contiguous order (first-index varies
        the fastest). If order is 'A', then the returned array may be in any order
        (either C-, Fortran-contiguous, or even discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned array
        will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have.
        Ones will be pre-pended to the shape as needed to meet this requirement.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.
    """
    return numpy.array(object, dtype=dtype, copy=copy, order=order, \
                       subok=subok, ndmin=ndmin, bohrium=bohrium)

def empty(shape, dtype=float, bohrium=True):
    """
    Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

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

    return numpy.empty(shape, dtype=dtype, bohrium=bohrium)

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
    A[:] = A.dtype.type(1)
    return A

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
    A[:] = A.dtype.type(0)
    return A

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

    if dtype == None:
        dtype = a.dtype
    if bohrium == None:
        bohrium = a.bohrium
    return empty(a.shape, dtype, bohrium)

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

    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[:] = b.dtype.type(0)
    return b

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

    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[:] = b.dtype.type(1)
    return b

def flatten(A):
    """
    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    a : array_like
        Array from which to retrive the flattened data from.

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> a = np.array([[1,2], [3,4]])
    >>> np.flatten(a)
    array([1, 2, 3, 4])
    """

    return A.reshape(numpy.multiply.reduce(numpy.asarray(A.shape)))

def diagonal(A,offset=0):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, a 1-D array containing the diagonal is returned.
        If the dimension of `a` is larger, then an array of diagonals is
        returned, "packed" from left-most dimension to right-most (e.g.,
        if `a` is 3-D, then the diagonals are "packed" along rows).

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : MATLAB work-a-like for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    """
    if A.ndim !=2 :
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (A.shape[0]-offset) > A.shape[1]:
            d = A[offset,:]
        else:
            d = A[offset:,0]
    else:
         if A.shape[1]-offset > A.shape[0]:
             d = A[:,offset]
         else:
             d = A[0,offset:]
    d.strides=(A.strides[0]+A.strides[1])
    return d

def diagflat(d,k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.

    Examples
    --------
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    d = numpy.asarray(d)
    d = flatten(d)
    size = d.size+abs(k)
    A = zeros((size,size), dtype=d.dtype, bohrium=d.bohrium)
    Ad = diagonal(A, offset=k)
    Ad[:] = d
    return A

def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.
    tril : Lower triange of an array.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """

    if v.ndim == 1:
        return diagflat(v,k)
    elif v.ndim == 2:
        return diagonal(v,k)
    else:
        raise ValueError("Input must be 1- or 2-d.")

def dot(a,b):
    """
    Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (without complex conjugation). For
    N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.

    Examples
    --------
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it's the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """
    if a.bohrium or b.bohrium:
        bridge.handle_array(a)
        bridge.handle_array(b)
    if b.ndim == 1:
        return numpy.add.reduce(a*b,-1)
    if a.ndim == 1:
        return add.reduce(a*numpy.transpose(b),-1)
    return add.reduce(a[:,numpy.newaxis]*numpy.transpose(b),-1)

def matmul(a,b):
    """
    Matrix multiplication of two 2-D arrays.

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the matrix multiplication of `a` and `b`.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    dot : Dot product of two arrays.

    Examples
    --------
    >>> np.matmul(np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]]))
    array([[19, 22],
           [43, 50]])
    """
    if a.dtype != b.dtype:
        raise ValueError("Input must be of same type")
    if a.ndim != 2 and b.ndim != 2:
        raise ValueError("Input must be 2-D.")
    if a.bohrium or b.bohrium:
        a.bohrium=True
        b.bohrium=True
        c = empty((a.shape[0],b.shape[1]),dtype=a.dtype)
        bridge.extmethod_exec("matmul",c,a,b)
        return c
    else:
    	return numpy.dot(a,b)

def fft(A):
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional discrete Fourier
    Transform (DFT).

    Parameters
    ----------
    A : array_like
        Input array, can be complex.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input transformed.

    See Also
    --------
    fft2 : The two-dimensional FFT.

    Examples
    --------
    >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([ -3.44505240e-16 +1.14383329e-17j,
             8.00000000e+00 -5.71092652e-15j,
             2.33482938e-16 +1.22460635e-16j,
             1.64863782e-15 +1.77635684e-15j,
             9.95839695e-17 +2.33482938e-16j,
             0.00000000e+00 +1.66837030e-15j,
             1.14383329e-17 +1.22460635e-16j,
             -1.64863782e-15 +1.77635684e-15j])

    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = np.fft.fft(np.sin(t))
    >>> freq = np.fft.fftfreq(t.shape[-1])
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part, as described in
    the `numpy.fft` documentation.
    """
    if A.bohrium and A.ndim <= 2:
      if A.dtype == numpy.complex64 or A.dtype == numpy.complex128: #maybe do type conversions for others
        B = empty(A.shape,dtype=A.dtype)
        bridge.fft(A,B)
        return B

	return numpy.fft.fft(A)

def fft2(A):
    """
    Compute the 2-dimensional discrete Fourier Transform

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    A : array_like
        Input array, can be complex

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    See Also
    --------
    fft : The one-dimensional FFT.

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.fft2(a)
    array([[  0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j],
           [  5.+0.j,   0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j],
           [ 10.+0.j,   0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j],
           [ 15.+0.j,   0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j],
           [ 20.+0.j,   0.+0.j,   0.+0.j,   0.+0.j,   0.+0.j]])

    """

    if A.bohrium and A.ndim == 2:
      if A.dtype == numpy.complex64 or A.dtype == numpy.complex128: #maybe do type conversions for others
        B = empty(A.shape,dtype=A.dtype)
        bridge.fft2(A,B)
        return B

	return numpy.fft.fft2(A)

def rad2deg(x, out=None):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Input array in radians.
    out : ndarray, optional
        Output array of same shape as x.

    Returns
    -------
    y : ndarray of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.

    See Also
    --------
    rad2deg : equivalent function

    Examples
    --------
    Convert a radian array to degrees

    >>> rad = np.arange(12.)*np.pi/6
    >>> np.degrees(rad)
    array([   0.,   30.,   60.,   90.,  120.,  150.,  180.,  210.,  240.,
            270.,  300.,  330.])

    >>> out = np.zeros((rad.shape))
    >>> r = degrees(rad, out)
    >>> np.all(r == out)
    True

    """

    if out == None:
        out = 180 * x / pi
    else:
        out[:] = 180 * x / pi
    return out

def deg2rad(x, out=None):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Angles in degrees.

    Returns
    -------
    y : ndarray
        The corresponding angle in radians.

    See Also
    --------
    rad2deg : Convert angles from radians to degrees.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    ``deg2rad(x)`` is ``x * pi / 180``.

    Examples
    --------
    >>> np.deg2rad(180)
    3.1415926535897931

    """

    if out == None:
        out = x * pi / 180
    else:
        out[:] = x * pi / 180
    return out

def logaddexp(x1, x2, out=None):
    """
    Logarithm of the sum of exponentiations of the inputs.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in
    statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : array_like
        Input values.

    Returns
    -------
    result : ndarray
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    logaddexp2: Logarithm of the sum of exponentiations of inputs in base-2.

    Examples
    --------
    >>> prob1 = np.log(1e-50)
    >>> prob2 = np.log(2.5e-50)
    >>> prob12 = np.logaddexp(prob1, prob2)
    >>> prob12
    -113.87649168120691
    >>> np.exp(prob12)
    3.5000000000000057e-50
    """

    if out == None:
        out = log(exp(x1) + exp(x2))
    else:
        out[:] = log(exp(x1) + exp(x2))
    return out

def logaddexp2(x1, x2, out=None):
    """
    Logarithm of the sum of exponentiations of the inputs in base-2.

    Calculates ``log2(2**x1 + 2**x2)``. This function is useful in machine
    learning when the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the base-2 logarithm of the calculated probability can be used instead.
    This function allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : array_like
        Input values.
    out : ndarray, optional
        Array to store results in.

    Returns
    -------
    result : ndarray
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    Examples
    --------
    >>> prob1 = np.log2(1e-50)
    >>> prob2 = np.log2(2.5e-50)
    >>> prob12 = np.logaddexp2(prob1, prob2)
    >>> prob1, prob2, prob12
    (-166.09640474436813, -164.77447664948076, -164.28904982231052)
    >>> 2**prob12
    3.4999999999999914e-50

    """

    if out == None:
        out = log2(exp2(x1) + exp2(x2))
    else:
        out[:] = log2(exp2(x1) + exp2(x2))
    return out

def modf(x, out1=None, out2=None):
    """
    Return the fractional and integral parts of an array, element-wise.

    The fractional and integral parts are negative if the given number is
    negative.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    y1 : ndarray
        Fractional part of `x`.
    y2 : ndarray
        Integral part of `x`.

    Notes
    -----
    For integer input the return values are floats.

    Examples
    --------
    >>> np.modf([0, 3.5])
    (array([ 0. ,  0.5]), array([ 0.,  3.]))
    >>> np.modf(-0.5)
    (-0.5, -0)

    """

    if out1 == None:
        out1 = mod(x,1.0)
    else:
        out1[:] = mod(x,1.0)
    if out2 == None:
        out2 = floor(x)
    else:
        out2[:] = floor(x)
    return (out1, out2)

def sign(x, out=None):
    """
    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.

    Parameters
    ----------
    x : array_like
      Input values.

    Returns
    -------
    y : ndarray
      The sign of `x`.

    Examples
    --------
    >>> np.sign([-5., 4.5])
    array([-1.,  1.])
    >>> np.sign(0)
    0

    """
    return add(multiply(less(x,0),-1),multiply(greater(x,0),1),out)

def signbit(x, out=None):
    """
    Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x: array_like
        The input value(s).
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved
        and it must be of the right shape to hold the output.
        See `doc.ufuncs`.

    Returns
    -------
    result : ndarray of bool
        Output array, or reference to `out` if that was supplied.

    Examples
    --------
    >>> np.signbit(-1.2)
    True
    >>> np.signbit(np.array([1, -2.3, 2.1]))
    array([False,  True, False], dtype=bool)

    """
    return less(x,0,out)

def hypot(x1, x2, out=None):
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.
    (See Examples)

    Parameters
    ----------
    x1, x2 : array_like
        Leg of the triangle(s).
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and it
        must be of the right shape to hold the output. See doc.ufuncs.

    Returns
    -------
    z : ndarray
        The hypotenuse of the triangle(s).

    Examples
    --------
    >>> np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> np.hypot(3*np.ones((3, 3)), [4])
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    """
    return sqrt(add(multiply(x1,x1),multiply(x2,bx2),out),out)


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
    if (not bohrium):
        return numpy.arange(start,stop,step,dtype)
    if (not stop):
        stop = start
        start = type(stop)(0)
    size = int(ceil((float(stop) - float(start)) / float(step)))
    if (dtype):
        start = dtype(start)
        stop  = dtype(stop)
        step  = dtype(step)
    else:
        dtype = int64
    return range(size,dtype=dtype) * step + start

def range(size, dtype=uint64):
    if (not isinstance(size, (int,long))):
        raise ValueError("size must be an integer")
    if (size < 1):
        raise ValueError("size must be greater than 0")
    if (dtype == int8 or
        dtype == int16 or
        dtype == int32 or
        dtype == uint8 or
        dtype == uint16 or
        dtype == uint32 or
        dtype == float16 or
        dtype == float32 or
        dtype == complex64):
        A = empty(size,dtype=uint32,bohrium=True)
    else:
        A = empty(size,dtype=uint64,bohrium=True)
    bridge.range(A)
    if (dtype != A.dtype.type):
        B = empty_like(A,dtype=dtype)
        B[:] = A[:]
        return B
    else:
        return A

def visualize(a, mode, colormap, min, max):
    if not (a.ndim == 2 or a.ndim == 3):
        raise ValueError("Input must be 2-D or 3-D.")
    if not a.bohrium:
        raise ValueError("Input must be a Bohrium array")
    if a.dtype == numpy.float32:
        raise ValueError("For now visualize only supports float32 arrays")

    if mode == "2d":
        flat = True
        cube = False
    elif mode == "3d":
        if a.ndim == 2:
            flat = False
            cube = False
        else:
            flat = False
            cube = True
    else:
        raise ValueError("Unknown mode '%s'"%mode)

    for s in a.shape:
        if s < 16:
            raise ValueError("Input shape must be greater than 15 element in each dimension")
    bridge.flush()#We will not delay the visualization
    args = array([float(colormap), float(flat), float(cube), float(min), float(max)], bohrium=True)
    bridge.extmethod_exec("visualizer",a,args,a)

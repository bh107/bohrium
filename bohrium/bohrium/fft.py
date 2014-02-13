"""
Discrete Fourier Transform (:mod:`bohrium.fft`)
=============================================

Background information
----------------------

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the signal from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency
domain*.

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as

.. math::
   A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}
   \\qquad k = 0,\\ldots,n-1.

The DFT is in general defined for complex inputs and outputs, and a
single-frequency component at linear frequency :math:`f` is
represented by a complex exponential
:math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`
is the sampling interval.

The values in the result follow so-called "standard" order: If ``A =
fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the mean of
the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
negative-frequency terms, in order of decreasingly negative frequency.
For an even number of input points, ``A[n/2]`` represents both positive and
negative Nyquist frequency, and is also purely real for real input.  For
an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
The routine ``np.fft.fftfreq(A)`` returns an array giving the frequencies
of corresponding elements in the output.  The routine
``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
that shift.

When the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
The phase spectrum is obtained by ``np.angle(A)``.

The inverse DFT is defined as

.. math::
   a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}
   \\qquad n = 0,\\ldots,n-1.

It differs from the forward transform by the sign of the exponential
argument and the normalization by :math:`1/n`.

Real and Hermitian transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the input is purely real, its transform is Hermitian, i.e., the
component at frequency :math:`f_k` is the complex conjugate of the
component at frequency :math:`-f_k`, which means that for real
inputs there is no information in the negative frequency components that
is not already available from the positive frequency components.
The family of `rfft` functions is
designed to operate on real inputs, and exploits this symmetry by
computing only the positive frequency components, up to and including the
Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
output points.  The inverses of this family assumes the same symmetry of
its input, and for an output of ``n`` points uses ``n/2+1`` input points.

Correspondingly, when the spectrum is purely real, the signal is
Hermitian.  The `hfft` family of functions exploits this symmetry by
using ``n/2+1`` complex points in the input (time) domain for ``n`` real
points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and
filtering.  The computational efficiency of the FFT means that it can
also be a faster way to compute large convolutions, using the property
that a convolution in the time domain is equivalent to a point-by-point
multiplication in the frequency domain.

In two dimensions, the DFT is defined as

.. math::
   A_{kl} =  \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}
   a_{mn}\\exp\\left\\{-2\\pi i \\left({mk\\over M}+{nl\\over N}\\right)\\right\\}
   \\qquad k = 0, \\ldots, N-1;\\quad l = 0, \\ldots, M-1,

which extends in the obvious way to higher dimensions, and the inverses
in higher dimensions also extend in the same way.

References
^^^^^^^^^^

.. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
        machine calculation of complex Fourier series," *Math. Comput.*
        19: 297-301.

.. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,
        2007, *Numerical Recipes: The Art of Scientific Computing*, ch.
        12-13.  Cambridge Univ. Press, Cambridge, UK.

Examples
^^^^^^^^

For examples, see the various functions.

"""

import bohrium as np
import numpy
import bohriumbridge as bridge

#Simular to fftn but support inverse fft as well
def _fftn(a, inverse=False):
    if not a.bohrium:
        raise ValueError("Input must be a Bohrium array")

    a = numpy.asarray(a, dtype=numpy.complex128)
    if not a.flags['C_CONTIGUOUS']:
        a = numpy.ascontiguousarray(a)
    b = np.empty_like(a)

    #FFTW uses sign to indicate inverse fft
    if inverse:
        sign = 1
    else:
        sign = -1

    args = np.array([numpy.int32(sign)], bohrium=True)
    bridge.extmethod_exec("fftw",b,a,args)
    return b


def fftn(a):
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    all axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex.

    Returns
    -------
    out : complex ndarray

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of all axes, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    See `numpy.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> np.fft.fftn(np.exp(2j * np.pi * np.arange(8) / 8))
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
    """
    return _fftn(a)

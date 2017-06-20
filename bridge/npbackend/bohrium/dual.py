# This module duplicates the NumPy API

from numpy_force.dual import *
from numpy_force.dual import register_func, restore_func, restore_all

__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2',
           'norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigvals',
           'eigh', 'eigvalsh', 'lstsq', 'pinv', 'cholesky', 'i0']

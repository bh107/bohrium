#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import bohrium as bh
from numpy.linalg import svd
import bohrium as bh
import util


def generate_gauss_matrix(filter_shape, sigma, data_type=np.float32):
    """
    Returns a Gauss filter matrix equivalent to the MATLAB
    fspecial('gaussian', dim, sigma) function

    Parameters
    ----------
    filter_shape : (int, int)
        Filter window shape (rows, cols) (must be odd)
    sigma : int
        The sigma value used to generate the `Gaussian window
        <http://en.wikipedia.org/wiki/Gaussian_filter>`_
    data_type : data-type, optional
        The precision of the generated gauss window

    Returns
    -------
    output : ndarray
        Gauss convolution matrix with the given dimensions, sigma and
        data_type.

    Raises
    ------
    ValueError
        If *filter_shape* values are even
    """

    if filter_shape[0] % 2 == 0 or filter_shape[1] % 2 == 0:
        msg = 'filter_shape: %s is _NOT_ odd' % str(filter_shape)
        raise ValueError(msg)

    result = np.zeros(filter_shape, dtype=data_type)

    y_radius = filter_shape[0] / 2
    x_radius = filter_shape[1] / 2

    for y in xrange(filter_shape[0]):
        y_distance = y - y_radius
        for x in xrange(filter_shape[1]):
            x_distance = x - x_radius
            result[y, x] = np.exp(-(y_distance ** 2 + x_distance ** 2) / (2.0 * sigma ** 2))

    result = result / result.sum()

    return result


def separate2d(input, data_type=np.float32):
    """
    `Separate a 2d-matrix convolution filter into two decomposed vectors
    <http://blogs.mathworks.com/steve/2006/11/28/separable-convolution-part-2/>`_

    Parameters
    ----------
    input : ndarray
        A 2-dimensional input array representing a convolution window
    out : Two tuple of 1-d ndarrays
        Tuple containing the two convolution vectors obtained by decomposing
        *input*

    Raises
    ------
    ValueError
        If *input* can't be decomposed into two vectors
    """

    # Singular Value Decomposition
    # NOTE: The convolution window is flipped:
    # http://www.songho.ca/dsp/convolution/convolution.html

    (U, S, V) = svd(input[::-1, ::-1])

    # Check rank of input matrix

    tolerance = max(input.shape) * np.spacing(max(S))
    rank = sum(S > tolerance)

    if rank != 1:
        msg = \
            'Decomposition error, \
             The number of linearly independent rows or columns are != 1'
        raise ValueError(msg)

    horizontal_vector = V[0] * np.sqrt(S[0])
    vertical_vector = U[:, 0] * np.sqrt(S[0])

    return (data_type(vertical_vector), data_type(horizontal_vector))


def zero_pad(data, window_vectors, bohrium=False):

    radius = (len(window_vectors[0]) / 2, len(window_vectors[1]) / 2)

    padded_data = bh.zeros((data.shape[0] + 2 * radius[0], data.shape[1] + 2 * radius[1]), dtype=data.dtype, bohrium=bohrium)

    padded_data[radius[0]:-radius[0], radius[1]:-radius[1]] = data

    return padded_data

def convolve2d_seperate(input, window_vectors, bohrium=False):

    window_radius = (len(window_vectors[0]) / 2, len(window_vectors[1])/ 2)

    out = bh.zeros(input.shape, dtype=input.dtype)

    padded_input = zero_pad(input, window_vectors, bohrium=bohrium)

    col_result = bh.zeros((input.shape[0], input.shape[1] + 2 * window_radius[1]), dtype=input.dtype, bohrium=bohrium)

    start_y = window_radius[0] * 2
    end_y = padded_input.shape[0]

    # First calculate dot product of image and Gauss vector
    # along columns (y direction) with radius 'tmp_window_radius[0]'
    # from input pixels

    for y in xrange(window_radius[0], input.shape[0] + window_radius[0]):
        start_y = y - window_radius[0]
        end_y = y + window_radius[0] + 1
        col_result[start_y] = bh.dot(padded_input[start_y:end_y].T, window_vectors[0][:,np.newaxis])[0]

    # Second calculate dot product of the dot products calculated above
    # and Gauss vector along rows (x direction)
    # with radius 'window_radius[1]' from input pixel

    for x in xrange(window_radius[1], input.shape[1] + window_radius[1]):
        start_x = x - window_radius[1]
        end_x = x + window_radius[1] + 1
        out[:, start_x] = bh.dot(col_result[:, start_x:end_x], window_vectors[1][:,np.newaxis])[0]
    return out

def main():
    B = util.Benchmark()

    data_type = np.float32
    input_size = int(B.size[0])
    input_shape = (input_size, input_size)
    filter_size = int(B.size[1])
    filter_shape = (filter_size, filter_size)

    input = np.arange(input_shape[0] * input_shape[1], dtype=data_type)

    # input[:] = random.random(input_shape[0]*input_shape[1])

    input.shape = input_shape

    sigma = 1

    filter = generate_gauss_matrix(filter_shape, sigma, data_type)
    filter = np.array(filter)
    (horizontal_vector, vectical_vector) = separate2d(filter, data_type)

    if B.bohrium:
        horizontal_vector = bh.array(horizontal_vector)
        vectical_vector = bh.array(vectical_vector)
        input = bh.array(input)

    print 'Convolve: %sx%s data with %sx%s filter'% (input_size, input_size, filter_size, filter_size)

    B.start()
    result = convolve2d_seperate(input, (horizontal_vector, vectical_vector), bohrium=B.bohrium)
    B.stop()
    B.pprint()



if __name__ == '__main__':
    main()


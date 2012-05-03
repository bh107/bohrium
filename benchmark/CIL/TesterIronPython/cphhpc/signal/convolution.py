#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# --- BEGIN_HEADER ---
#
# convolution - http://www.songho.ca/dsp/convolution/convolution.html
# Copyright (C) 2011-2012  The CPHHPC Project lead by Brian Vinter
#
# This file is part of CPHHPC Toolbox.
#
# CPHHPC Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CPHHPC Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
#
# -- END_HEADER ---
#

"""Convolution: http://www.songho.ca/dsp/convolution/convolution.html"""
from numcil import zeros

def convolve2d(input, window, out=None, data_type=None):
    """
    Convolve two 2-dimensional arrays:
    http://www.songho.ca/dsp/convolution/convolution.html
    
    Parameters
    ----------
    input : ndarray
        A 2-dimensional input array
    window: ndarray
        A 2-dimensional convolution window array (shape must be odd)
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right shape and must be
        C-contiguous. This is a performance feature. Therefore, if
        these conditions are not met, an exception is raised, instead of
        attempting to be flexible.
    data_type : data-type, optional
        The precision of the created `out` ndarray if `out` is None
    Raises
    ------
    ValueError
        If shape of `window` is even
        If shape of `out` doesn't match those of `input`
        
    """
    if window.shape[0] % 2 == 0 or window.shape[1] % 2 == 0:
        msg = "window.shape: %s is _NOT_ odd" % (str(window.shape))
        raise ValueError(msg)
    
    window_radius = (window.shape[0]/2, window.shape[1]/2)
    
    zero_pad_shape = (input.shape[0] + (window_radius[0]*2),
                      input.shape[1] + (window_radius[1]*2))
    
    zero_padded_input = zeros(zero_pad_shape, dtype=data_type)

    zero_padded_input[window_radius[0]:-window_radius[0],
                      window_radius[1]:-window_radius[1]] = input

    if out != None:
        if out.shape != input.shape:
            msg = "input.shape: %s and out.shape: %s doesn't match" % (str(input.shape), str(out.shape))
            raise ValueError(msg)
    else:
        if data_type == None:
            out = zeros(input.shape, dtype=input.dtype)
        else:
            out = zeros(input.shape, dtype=data_type)
        
    start_y = window_radius[0]*2
    end_y = zero_pad_shape[0]
    
    for y in xrange(window.shape[0]):
        start_x = window_radius[1]*2
        end_x = zero_pad_shape[1]

        for x in xrange(window.shape[1]):
            tmp = zero_padded_input * window[y][x]
            out += tmp[start_y:end_y, start_x:end_x]
            start_x -= 1
            end_x -= 1
            
        start_y -= 1
        end_y -= 1

    return out


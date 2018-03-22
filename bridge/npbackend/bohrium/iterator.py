import copy

from .bhary import fix_biclass_wrapper
from . import array_create

class iterator(object):
    '''Iterator used for sliding views within loops.

    Notes
    -----
    Supports addition, subtraction and multiplication.
    '''

    def __init__(self, value):
        self.step = 1
        self.offset = value
        self.max_iter = 0

    def __add__(self, other):
        new_it = copy.copy(self)
        new_it.offset += other
        return new_it

    def __radd__(self, other):
        new_it = copy.copy(self)
        new_it.offset += other
        return new_it

    def __sub__(self, other):
        new_it = copy.copy(self)
        new_it.offset -= other
        return new_it

    def __rsub__(self, other):
        new_it = copy.copy(self)
        new_it.step *= -1
        new_it.offset = other - new_it.offset
        return new_it

    def __mul__(self, other):
        new_it = copy.copy(self)
        new_it.offset *= other
        new_it.step *= other
        return new_it

    def __rmul__(self, other):
        new_it = copy.copy(self)
        new_it.offset *= other
        new_it.step *= other
        return new_it

    def __neg__(self):
        new_it = copy.copy(self)
        new_it.step *= -1
        new_it.offset *= -1
        return new_it

class IteratorOutOfBounds(Exception):
    '''Exception thrown when a view goes out of bounds after the maximum
       iterations.'''
    def __init__(self, dim, shape, first_index, last_index):
        error_msg = "\n    Iterator out of bounds:\n" \
                    "     Dimension %d has length %d, iterator starts from %d and goes to %d." \
                    % (dim, shape, first_index, last_index)
        super(IteratorOutOfBounds, self).__init__(error_msg)


class ViewShape(Exception):
    '''Exception thrown when a view changes shape between iterations in a
       loop.'''
    def __init__(self, start,stop):
        error_msg = "\n    View must not change shape between iterations:\n" \
                    "    Stride of view start is %d, stride of view end is %d." \
                    % (start, stop)
        super(ViewShape, self).__init__(error_msg)


def get_iterator(max_iter, val):
    '''Returns an iterator with a given starting value. An iterator behaves like
       an integer, but is used to slide view between loop iterations.

    Parameters
    ----------
    max_iter : int !! eller none? !!
        The maximum amount of iterations of the loop. Used for checking
        boundaries.
    val : int
        The initial value of the iterator.

    Notes
    -----
    `get_iterator` can only be used within a bohrium loop function. Within the
    loop `max_iter` is set by a lambda function. This is also the case in the
    example.

    Examples
    --------
    >>> def kernel(a):
    ...     i = get_iterator(1)
    ...     a[i] *= a[i-1]
    >>> a = bh.arange(1,6)
    >>> bh.do_while(kernel, 4, a)
    array([1, 2, 6, 24, 120])'''

    it = iterator(val)
    setattr(it, 'max_iter', max_iter)
    return it


@fix_biclass_wrapper
def has_iterator(*s):
    '''Checks whether a (multidimensional) slice contains an iterator

    Parameters
    ----------
    s : pointer to an integer, iterator or a tuple of integers/iterators

    Notes
    -----
    Only called from __getitem__ in bohrium arrays (see _bh.c) and .'''

    # Helper function for one-dimensional slices
    def check_simple_type(ss):
        if isinstance(ss, slice):
            # Checks whether the view changes shape during iterations
            return isinstance(ss.start, iterator) or \
                   isinstance(ss.stop, iterator)
        else:
            return isinstance(ss, iterator)
    # Checking single or multidimensional slices for iterators
    if isinstance(s, tuple):
        for t in s:
            it = check_simple_type(t)
            if it: return it
        return False
    else:
        return check_simple_type(s)


@fix_biclass_wrapper
def slide_from_view(a, sliced):
    def check_bounds(shape, dim, s):
        '''Checks whether the view is within the bounds of the array,
        given the maximum number of iterations'''
        last_index = s.offset + s.step * (s.max_iter-1)
        if -shape[dim] <= s.offset   < shape[dim] and \
           -shape[dim] <= last_index < shape[dim]:
            return True
        else:
            raise IteratorOutOfBounds(dim, shape[dim], s.offset, last_index)


    def check_shape(s):
        '''Checks whether the view changes shape between iterations.'''
        if isinstance(s.start, iterator) and \
           isinstance(s.stop, iterator):
            if s.start.step != s.stop.step:
                raise ViewShape(s.start.step, s.stop.step)
        elif isinstance(s.start, iterator):
            raise ViewShape(s.start.step, 0)
        elif isinstance(s.stop, iterator):
            raise ViewShape(0, s.stop.step)
        return True


    if not isinstance(sliced, tuple):
        sliced = (sliced,)

    new_slices = ()
    slides = []
    for i, s in enumerate(sliced):
        if len(sliced) == 1 or has_iterator(s):
            # A slice with optional step size (eg. a[i:i+2] or a[i:i+2:2])
            if isinstance(s, slice):
                check_shape(s)
                if s.step:
                    # Set the correct step size
                    setattr(s.start, "step", s.start.step*s.step)
                    setattr(s.stop, "step", s.stop.step*s.step)
                    # Check whether the iterator stays within the array
                check_bounds(a.shape, i, s.start)
                check_bounds(a.shape, i, s.stop-1)
                new_slices += (slice(s.start.offset, s.stop.offset),)
                slides.append((i, s.start.step))

            # A single iterator (eg. a[i])
            else:
                # Check whether the iterator stays within the array
                check_bounds(a.shape, i, s)
                if s.offset == -1:
                    new_slices += (slice(s.offset, None),)
                else:
                    new_slices += (slice(s.offset, s.offset+1),)
                slides.append((i, s.step))
        else:
            new_slices += (s,)
    return slide_view(a, new_slices, slides)


def slide_view(a, s, dim_stride_tuples):
    """Creates a dynamic view within a loop, that updates the given dimensions
       by the given strides at the end of each iteration.

    Parameters
    ----------
    a : array view
        A view into an array
    dim_stride_tuples: (int, int)[]
        A list of (dimension, stride) pairs. For each of these pairs, the
        dimension is updated by the stride in each iteration of a loop."""
    from . import _bh

    # Allocate a new view
    b = a[s]

    # Set the relevant update conditions for the new view
    for (dim, stride) in dim_stride_tuples:
        _bh.slide_view(a, b, dim, stride)
    return b

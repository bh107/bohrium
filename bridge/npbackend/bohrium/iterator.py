import copy

from .bhary import fix_biclass_wrapper
from . import array_create
from functools import reduce

class iterator(object):
    '''Iterator used for sliding views within loops.

    Notes
    -----
    Supports addition, subtraction and multiplication.
    '''

    def __init__(self, max_iter, value, step_delay=1, reset=None):
        '''The initial state of the iterator.

        Parameters
        ----------
        max_iter : int
            The maximum amount of iterations the loop can go on for
        value : int
            The beginning offset of the iterator
        step_delay : int
            The amount of iterations needed before a change is made
        reset : int
            The amount of iterations before the changes are reset

        Notes
        -----
        step : int
            The amount the offset is slided when performing a step

        The step can be changed by using multiplication
        '''

        self.max_iter = max_iter
        self.offset = value
        self.step_delay = step_delay
        self.reset = reset

        self.step = 1

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
        error_msg = \
            "\n    Iterator out of bounds:\n" \
            "     Dimension %d has length %d, iterator starts from %d and goes to %d." \
                    % (dim, shape, first_index, last_index)
        super(IteratorOutOfBounds, self).__init__(error_msg)


class dynamic_view_info(object):
    '''Object for storing information about dynamic changes to the view'''
    def __init__(self, dynamic_changes, shape, stride, reset=[]):
        '''The initial state of the dynamic view information.

        Parameters
        ----------
        dynamic_changes : [(int,int,int)] or [(int,int,int,int)]
            A list of tuples corresponding to the dynamic changes the view.
            The tuple is (dimension, slide, shape_change, step_delay).
            If the step delay is not explicitly stated, it is set to 1.
        shape : int
            The shape of the view that the dynamic view is based upon.
            Used when using negative indices to wrap around.
        stride : int
            The stride of the view that the dynamic view is based upon.
            Used for making slides to the offset.
        reset : [(int, int)]
            A list of tuples corresponding to the dimension and
            amount of iterations before the changes are reset.
            Used for nested loops.
        '''

        # Shape and stride of the view that the dynamic view is based upon
        self.shape = shape
        self.stride = stride

        assert(len(dynamic_changes[0]) == 3 or len(dynamic_changes[0]) == 4)
        if len(dynamic_changes[0]) == 3:
            # Dynamic changes without explicit step delay
            # Format: [(dimension, slide, shape)]
            self.dynamic_changes = []
            for (dim, slide, shape_change) in dynamic_changes:
                self.dynamic_changes.append((dim, slide, shape_change, 1))
        else:
            # Dynamic changes with explicit step delay
            # Format: [(dimension, slide, shape, step_delay)]
            self.dynamic_changes = dynamic_changes
        self.resets = reset


def inherit_dynamic_changes(a, sliced):
    '''Creates a view into another view which has dynamic changes.
    The new view inherits the dynamic changes.'''
    # Temporary store the dynamic changes
    dvi = a.bhc_dynamic_view_info

    # Perform slicing (removes the dynamic changes to avoid infinite recursion)
    a.bhc_dynamic_view_info = None
    b = a[sliced]

    # Inherit the dynamic changes (and restore dynamic changes to a)
    b.bhc_dynamic_view_info = dvi
    a.bhc_dynamic_view_info = dvi
    return b


def get_iterator(max_iter, val, step_delay=1):
    '''Returns an iterator with a given starting value. An iterator behaves like
       an integer, but is used to slide view between loop iterations.

    Parameters
    ----------
    max_iter : int
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

    it = iterator(max_iter, val, step_delay)
    return it


def get_grid(max_iter, *args):
    '''Returns n iterators in a grid, corresponding to nested loops.

    Parameters
    ----------
    args : pointer to two or more integers
        The first integer is the maximum iterations of the loop, used for checking
        boundaries. The rest are the shape of the grid.

    Notes
    -----
    `get_grid` can only be used within a bohrium loop function. Within the
    loop `max_iter` is set by a lambda function.
    There are no upper bound on the amount of grid values.

    Examples
    --------
    >>> def kernel(a):
    ...     i, j, k = get_grid(3,3,3)
    ...     a[i,j,k] += 1

    correspondes to

    >>> for i in range(3):
    ...     for j in range(3):
    ...         for k in range(3):
    ...             a[i,j,k] += 1'''

    # Maximum iterations of the loop
#    max_iter = args[0]

    # Remove maximum iterations and reverse the grid to
    # loop over the grid from inner to outer
    grid = args[::-1]
    # Tuple of resulting iterators
    iterators = ()

    # Beginning step delay is always 1
    step_delay = 1

    for dim, iterations in enumerate(grid):
        i = iterator(max_iter, 0, step_delay, iterations)
        step_delay *= iterations
        iterators = (i,) + iterators
    return iterators


@fix_biclass_wrapper
def has_iterator(*s):
    '''Checks whether a (multidimensional) slice contains an iterator

    Parameters
    ----------
    s : pointer to an integer, iterator or a tuple of integers/iterators

    Notes
    -----
    Only called from __getitem__ and __setitem__ in bohrium arrays (see _bh.c).'''

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
            if it:
                return it
        return False
    else:
        return check_simple_type(s)


@fix_biclass_wrapper
def slide_from_view(a, sliced):
    def check_bounds(shape, dim, s):
        '''Checks whether the view is within the bounds of the array,
        given the maximum number of iterations'''

        # If the dimension is reset, then only the range before the reset needs to be checked
        if s.reset and s.max_iter / s.step_delay >= s.reset:
            last_index = s.offset + (s.reset-1) * s.step
        else:
            last_index = s.offset + s.step / s.step_delay * (s.max_iter-1)

        # Check that the starting index and the last index is within bounds
        if -shape[dim] <= s.offset   < shape[dim] and \
           -shape[dim] <= last_index < shape[dim]:
            return True
        else:
            raise IteratorOutOfBounds(dim, shape[dim], s.offset, last_index)

    def dynamic_shape_change(s):
        '''Returns how the shape of a view changes between iterations
        based on a slice possibly containing iterators'''
        if isinstance(s.start, iterator):
            start_step = s.start.step
        else:
            start_step = 0
        if isinstance(s.stop, iterator):
            stop_step = s.stop.step
        else:
            stop_step = 0
        return stop_step - start_step

    # Make sure that the indices is within a tuple
    if not isinstance(sliced, tuple):
        sliced = (sliced,)

    # Checks whether the indices contains a slice
    has_slices = reduce((lambda x, y: x or y), [isinstance(s, slice) for s in sliced])

    # The new slices (does not contain iterators)
    new_slices = ()
    # The dynamic changes
    slides = []
    # The resets (used for resetting an iterator in nested loops)
    resets = []

    for i, s in enumerate(sliced):
        if len(sliced) == 1 or has_iterator(s):
            # A slice with optional step size (eg. a[i:i+2] or a[i:i+2:2])
            if isinstance(s, slice):
                start_is_iterator = isinstance(s.start, iterator)
                stop_is_iterator = isinstance(s.stop, iterator)

                # Cannot contain iterators with different reset in same slice
                if start_is_iterator and stop_is_iterator \
                   and not s.start.reset == s.stop.reset:
                    raise("The iterators within a single slice must be at the same depth of the grid")

                # Check whether the start/end iterator stays within the array
                if start_is_iterator:
                    check_bounds(a.shape, i, s.start)
                    start = s.start.offset
                    step = s.start.step
                    step_delay = s.start.step_delay
                    reset = s.start.reset
                else:
                    start = s.start
                    step = 0
                    step_delay = 1

                if stop_is_iterator:
                    check_bounds(a.shape, i, s.stop)
                    stop = s.stop.offset
                    reset = s.stop.reset
                else:
                    stop = s.stop

                # Store the new slice
                new_slices += (slice(start, stop, s.step),)
                slides.append((i, step, dynamic_shape_change(s), step_delay))

            # A single iterator (eg. a[i])
            else:
                # Check whether the iterator stays within the array
                check_bounds(a.shape, i, s)

                # If the indices does not contain a slice, the returned value must
                # be a view with shape 1 in each dimension (To avoid a flush)
                if not has_slices:
                    new_slices += (slice(s.offset, s.offset+1),)
                else:
                    new_slices += (s.offset,)
                slides.append((i, s.step, 0, s.step_delay))

                reset = s.reset

            # Add information about dimension being reset
            if reset:
                resets.append((i,s.reset))
        else:
            # Does not contain an iterator, just pass it through
            new_slices += (s,)

    # Use the indices to create a new view
    b = a[new_slices]

    # If the view, which is indexed into, contains dynamic changes,
    # pass them on to the new view
    a_dvi = a.bhc_dynamic_view_info
    if a_dvi:
        o_shape = a_dvi.shape
        o_stride = a_dvi.stride
        o_dynamic_changes = a_dvi.dynamic_changes
        new_stride = [(b.strides[i] / a.strides[i]) for i in range(b.ndim)]
        new_slides = []
        for (b_dim, b_slide, b_shape_change, b_step_delay) in slides:
            for (a_dim, a_slide, _, a_step_delay) in o_dynamic_changes:
                if a_step_delay != b_step_delay:
                    raise("A view cannot use iterators from other depths of the grid, than the view it is based upon.")
                if a_dim == b_dim:
                    parent_stride = a.strides[a_dim] / o_stride[a_dim]
                    b_slide = a_slide + b_slide * parent_stride
            new_slides.append((b_dim, b_slide, b_shape_change, b_step_delay))
        dvi = dynamic_view_info(new_slides, o_shape, o_stride)
    else:
        dvi = dynamic_view_info(slides, a.shape, a.strides)

    dvi.resets = resets
    b.bhc_dynamic_view_info = dvi
    return b

def add_slide_info(a):
    """Checks whether a view is dynamic and adds the relevant
       information to the view structure within BXX if it is.

    Parameters
    ----------
    a : array view
        A view into an array
    """
    from . import _bh

    # Check whether the view is a dynamic view
    dvi = a.bhc_dynamic_view_info

    if dvi:
        # Set the relevant update conditions for the new view
        for (dim, slide, shape_change, step_delay) in dvi.dynamic_changes:
            # Stride is bytes, so it has to be diveded by 8
            try:
                stride = dvi.stride[dim]/8
                shape = dvi.shape[dim]
            except:
                stride = 0
                shape = 0
            # Add dynamic information to the view within the cxx bridge
            _bh.slide_view(a, dim, slide, shape_change, shape, stride, step_delay)

        # Add resets to the relevant dimensions within the cxx bridge (used for nested loops)
        for (dim,reset_max) in dvi.resets:
            _bh.add_reset(a, dim, reset_max)

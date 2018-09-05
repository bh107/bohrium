"""
Bohrium Loop
============
"""

import sys
import copy
from functools import reduce
import numpy_force as numpy
from . import bhary


class Iterator(object):
    """Iterator used for sliding views within loops.

    Notes
    -----
    Supports addition, subtraction and multiplication.
    """

    def __init__(self, max_iter, value, step_delay=1, reset=None):
        """The initial state of the iterator.

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
        """

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
    """Exception thrown when a view goes out of bounds after the maximum
       iterations."""

    def __init__(self, dim, shape, first_index, last_index):
        error_msg = \
            "\n    Iterator out of bounds:\n" \
            "     Dimension %d has length %d, iterator starts from %d and goes to %d." \
            % (dim, shape, first_index, last_index)
        super(IteratorOutOfBounds, self).__init__(error_msg)


class IteratorIllegalDepth(Exception):
    """Exception thrown when a view consists of a mix of iterator depths."""

    def __init__(self):
        error_msg = \
            "\n    Illegal mix of iterators with different depth:\n" \
            "     A view cannot use iterators from different depths of the grid in the same dimension."
        super(IteratorIllegalDepth, self).__init__(error_msg)


class IteratorIllegalBroadcast(Exception):
    """Exception thrown when a view consists of a mix of iterator depths."""

    def __init__(self, dim, a_shape, a_shape_change,
                 bcast_shape, bcast_shape_change):
        error_msg = \
            "Broadcast with dynamic shape:\n" \
            "    View with shape " + str(a_shape) + \
            " changes shape by " + str(a_shape_change) + \
            " in dimension " + str(dim) + \
            ".\n    It is differet from the view it is broadcasted from with shape " + \
            str(bcast_shape) + \
            " which changes shape by " + str(bcast_shape_change) + ".\n"
        super(IteratorIllegalBroadcast, self).__init__(error_msg)


class DynamicViewInfo(object):
    """Object for storing information about dynamic changes to the view"""

    def __init__(self, dynamic_changes, shape, stride, resets={}):
        """The initial state of the dynamic view information.

        Parameters
        ----------
        dynamic_changes : {int : [(int, int, int, int, int)]}
            A dictionary of lists of tuples corresponding to the dynamic
            changes the view.
            The tuple is (slide, shape_change, step_delay, shape, stride)
            and is explained further in the parameters to `add_dynamic_change`.
        shape : int tuple (variable length)
            The shape of the view that the dynamic view is based upon.
            Used when using negative indices to wrap around.
        stride : int tuple (variable length)
            The stride of the view that the dynamic view is based upon.
            Used for making slides to the offset.
        resets : {int : int}
            A dictionary of ints. The key corresponds to the dimension and
            the value to iterations before the changes are reset.
            Used for nested loops.
        """
        self.shape = shape
        self.stride = stride
        self.dynamic_changes = dynamic_changes
        self.resets = resets

    def add_dynamic_change(self, dim, slide, shape_change, step_delay, shape=None, stride=None):
        """Add dynamic changes to the dynamic changes information of the view.

        Parameters
        ----------
        dim : int
            The relevant dimension
        slide : int
            The change to the offset in the given dimension
            (can be both positive and negative)
        shape_change : int
            The amount the shape changes in the given dimension
            (can also be both positive and negative)
        step_delay : int
            If the change is based on an iterator in a grid, it is
            the changes can be delayed until the inner iterators
            have been updated `step_delay` times.
        shape : int
            The shape that the view can slide within. If not given,
            self.shape[dim] is used instead
        stride : int
            The stride that the view can slide within. If not given,
            self.stride[dim] is used instead
        """
        if not shape:
            shape = self.shape[dim]
        if not stride:
            stride = self.stride[dim]

        if self.has_changes_in_dim(dim):
            self.dynamic_changes[dim].append((slide, shape_change, step_delay, shape, stride))
        else:
            self.dynamic_changes[dim] = [(slide, shape_change, step_delay, shape, stride)]

    def has_changes_in_dim(self, dim):
        """Check whether there are any dynamic changes in the given dimension.

        Parameters
        ----------
        dim : int
            The relevant dimension
        """
        return dim in self.dynamic_changes

    def dims_with_changes(self):
        """Returns a list of all dimensions with dynamic changes."""
        return self.dynamic_changes.keys()

    def changes_in_dim(self, dim):
        """Returns a list of all dynamic changes in a dimension.
        If the dimension does not contain any dynamic changes,
        an empty list is returned.

        Parameters
        ----------
        dim : int
            The relevant dimension
        """
        changes = []
        if self.has_changes_in_dim(dim):
            changes += self.dynamic_changes[dim]
        return changes

    def index_into(self, dvi):
        """Modifies the dynamic change such that is reflects
        being indexed into another view with dynamic changes.

        Parameters
        ----------
        dim : DynamicViewInfo
            The infomation about dynamic changes within the
            view which is indexed into
        """
        for dim in dvi.dims_with_changes():
            if (dim in self.resets) or (dim in dvi.resets):
                if not (dim in dvi.resets) or \
                        not (dim in self.resets) or \
                        self.resets[dim] != dvi.resets[dim] or \
                        self.changes_in_dim(dim) != dvi.changes_in_dim(dim):
                    raise IteratorIllegalDepth()
            for change in dvi.changes_in_dim(dim):
                self.add_dynamic_change(dim, *change)

    def dim_shape_change(self, dim):
        """Returns the summed shape change in the given dimension.

        Parameters
        ----------
        dim : int
            The relevant dimension
        """
        shape_change_sum = 0
        if self.has_changes_in_dim(dim):
            for (_, shape_change, _, _, _) in self.dynamic_changes[dim]:
                shape_change_sum += shape_change
        return shape_change_sum

    def get_shape_changes(self):
        """Returns a dictionary of all changes to the shape.
        The dimension is the key and the shape change in the
        dimension is the value.
        """
        shape_changes = {}
        for dim in self.dynamic_changes.keys():
            shape_changes[dim] = 0
            for (_, shape_change, _, _, _) in self.dynamic_changes[dim]:
                shape_changes[dim] += shape_change
        return shape_changes

    def has_changes(self):
        """Returns whether the object contains any dynamic changes."""
        return self.dynamic_changes != {}


def inherit_dynamic_changes(a, sliced):
    """Creates a view into another view which has dynamic changes.
    The new view inherits the dynamic changes."""
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
    """Returns an iterator with a given starting value. An iterator behaves like
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
    array([1, 2, 6, 24, 120])"""

    it = Iterator(max_iter, val, step_delay)
    return it


def get_grid(max_iter, *args):
    """Returns n iterators in a grid, corresponding to nested loops.

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
    ...             a[i,j,k] += 1"""

    # Remove maximum iterations and reverse the grid to
    # loop over the grid from inner to outer
    grid = args[::-1]

    # Tuple of resulting iterators
    iterators = ()

    # Beginning step delay is always 1
    step_delay = 1

    for dim, iterations in enumerate(grid):
        i = Iterator(max_iter, 0, step_delay, iterations)
        step_delay *= iterations
        iterators = (i,) + iterators
    return iterators


def has_iterator(*s):
    """Checks whether a (multidimensional) slice contains an iterator

    Parameters
    ----------
    s : pointer to an integer, iterator or a tuple of integers/iterators

    Notes
    -----
    Only called from __getitem__ and __setitem__ in bohrium arrays (see _bh.c)."""

    # Helper function for one-dimensional slices
    def check_simple_type(ss):
        if isinstance(ss, slice):
            # Checks whether the view changes shape during iterations
            return isinstance(ss.start, Iterator) or \
                   isinstance(ss.stop, Iterator)
        else:
            return isinstance(ss, Iterator)

    # Checking single or multidimensional slices for iterators
    if isinstance(s, tuple):
        for t in s:
            it = check_simple_type(t)
            if it:
                return it
        return False
    else:
        return check_simple_type(s)


def slide_from_view(a, sliced):
    def check_bounds(shape, dim, s):
        """Checks whether the view is within the bounds of the array,
        given the maximum number of iterations"""

        # If the dimension is reset, then only the range before the reset needs to be checked
        if s.reset and s.max_iter / s.step_delay >= s.reset:
            last_index = s.offset + (s.reset - 1) * s.step
        else:
            last_index = s.offset + s.step / s.step_delay * (s.max_iter - 1)

        # Check that the starting index and the last index is within bounds
        if -shape[dim] <= s.offset < shape[dim] and \
                -shape[dim] <= last_index < shape[dim]:
            return True
        else:
            raise IteratorOutOfBounds(dim, shape[dim], s.offset, last_index)

    def dynamic_shape_change(s):
        """Returns how the shape of a view changes between iterations
        based on a slice possibly containing iterators"""
        if isinstance(s.start, Iterator):
            start_step = s.start.step
        else:
            start_step = 0
        if isinstance(s.stop, Iterator):
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
    resets = {}

    for i, s in enumerate(sliced):
        if len(sliced) == 1 or has_iterator(s):
            # A slice with optional step size (eg. a[i:i+2] or a[i:i+2:2])
            if isinstance(s, slice):
                start_is_iterator = isinstance(s.start, Iterator)
                stop_is_iterator = isinstance(s.stop, Iterator)

                # Cannot contain iterators with different reset in same slice
                if start_is_iterator and stop_is_iterator \
                        and (s.start.step_delay != s.stop.step_delay or s.start.reset != s.stop.reset):
                    raise IteratorIllegalDepth()

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
                    if s.stop.offset > 0:
                        check_bounds(a.shape, i, s.stop - 1)
                    else:
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
                    new_slices += (slice(s.offset, s.offset + 1),)
                else:
                    new_slices += (s.offset,)
                slides.append((i, s.step, 0, s.step_delay))
                reset = s.reset

            # Add information about dimension being reset
            if reset:
                resets[i] = reset
        else:
            # Does not contain an iterator, just pass it through
            new_slices += (s,)

    # Use the indices to create a new view
    b = a[new_slices]

    a_dvi = a.bhc_dynamic_view_info

    if a_dvi:
        b_dvi = DynamicViewInfo({}, a_dvi.shape, a.strides, resets)
    else:
        b_dvi = DynamicViewInfo({}, a.shape, a.strides, resets)

    for slide in slides:
        b_dvi.add_dynamic_change(*slide)

    # If the view, which is indexed into, contains dynamic changes,
    # pass them on to the new view

    if a_dvi:
        b_dvi.index_into(a_dvi)

    if b_dvi.has_changes():
        b.bhc_dynamic_view_info = b_dvi
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
        for dim in dvi.dynamic_changes.keys():
            for (slide, shape_change, step_delay, shape, stride) in dvi.dynamic_changes[dim]:
                try:
                    stride = int(stride / 8)
                    shape = shape
                except:
                    stride = 0
                    shape = 0

                # Add dynamic information to the view within the cxx bridge
                _bh.slide_view(a, dim, slide, shape_change, shape, stride, step_delay)
        # Add resets to the relevant dimensions within the cxx bridge (used for nested loops)
        for dim in dvi.resets.keys():
            _bh.add_reset(a, dim, dvi.resets[dim])


def do_while(func, niters, *args, **kwargs):
    """Repeatedly calls the `func` with the `*args` and `**kwargs` as argument.

    The `func` is called while `func` returns True or None and the maximum number
    of iterations, `niters`, hasn't been reached.

    Parameters
    ----------
    func : function
        The function to run in each iterations. `func` can take any argument and may return
        a boolean `bharray` with one element.
    niters: int or None
        Maximum number of iterations in the loop (number of times `func` is called). If None, there is no maximum.
    *args, **kwargs : list and dict
        The arguments to `func`

    Notes
    -----
    `func` can only use operations supported natively in Bohrium.

    Examples
    --------
    >>> def loop_body(a):
    ...     a += 1
    >>> a = bh.zeros(4)
    >>> bh.do_while(loop_body, 5, a)
    >>> a
    array([5, 5, 5, 5])

    >>> def loop_body(a):
    ...     a += 1
    ...     return bh.sum(a) < 10
    >>> a = bh.zeros(4)
    >>> bh.do_while(loop_body, None, a)
    >>> a
    array([3, 3, 3, 3])
    """

    from . import _bh
    _bh.flush()
    flush_count = _bh.flush_count()
    func.__globals__['get_iterator'] = lambda x=0: get_iterator(niters, x)
    func.__globals__['get_grid'] = lambda *argz: get_grid(*((niters,) + argz))
    cond = func(*args, **kwargs)
    if flush_count != _bh.flush_count():
        raise TypeError("Invalid `func`: the looped function contains operations not support "
                        "by Bohrium, contain branches, or is simply too big!")
    if niters is None:
        niters = sys.maxsize - 1
    if cond is None:
        _bh.flush_and_repeat(niters, None)
    else:
        if not bhary.check(cond):
            raise TypeError("Invalid `func`: `func` may only return Bohrium arrays or nothing at all")
        if cond.dtype.type is not numpy.bool_:
            raise TypeError("Invalid `func`: `func` returned array of wrong type `%s`. "
                            "It must be of type `bool`." % cond.dtype)
        if len(cond.shape) != 0 and len(cond) > 1:
            raise TypeError("Invalid `func`: `func` returned array of shape `%s`. "
                            "It must be a scalar or an array with one element." % cond.shape)
        if not bhary.is_base(cond):
            raise TypeError("Invalid `func`: `func` returns an array view. It must return a base array.")

        _bh.sync(cond)
        _bh.flush_and_repeat(niters, cond)

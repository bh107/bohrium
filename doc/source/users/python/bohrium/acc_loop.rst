Accelerate Loops
~~~~~~~~~~~~~~~~

As we all know, having for and while loops in Python is bad for performance but is sometimes necessary.  E.g. in the case of the ``heat2d()`` code, we have to evaluate ``delta > epsilon`` in order to know when to stop iterating. To address this issue, Bohrium introduces the function :func:`bohrium.loop.do_while()`, which takes a function and calls it repeatedly until either a maximum number of calls has been reached or until the function return False.

The function signature::

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
        """

An example where the function doesn't return anything::

        >>> def loop_body(a):
        ...     a += 1
        >>> a = bh.zeros(4)
        >>> bh.do_while(loop_body, 5, a)
        >>> a
        array([5, 5, 5, 5])

An example where the function returns a ``bharray`` with one element and of type ``bh.bool``::

        >>> def loop_body(a):
        ...     a += 1
        ...     return bh.sum(a) < 10
        >>> a = bh.zeros(4)
        >>> bh.do_while(loop_body, None, a)
        >>> a
        array([3, 3, 3, 3])


Sliding Views Between Iterations
--------------------------------

It can be useful to increase/decrease the beginning of certain array views between iterations of a loop. This can be achieved using :func:`bohrium.loop.get_iterator()`, which returns a special bohrium iterator. The iterator can be given an optional start value (0 by default). The iterator is increased by one for each iteration, but can be changed increase or decrease by multiplying any constant (see example 2).

Iterators only supports addition, subtraction and multiplication. :func:`bohrium.loop.get_iterator()` can only be used within Bohrium loops. Views using iterators cannot change shape between iterations. Therefore, views such as ``a[i:2*i]`` are not supported.

Example 1. Using iterators to create a loop-based function for calculating the triangular numbers (from 1 to 10). The loop in numpy looks the following::

        >>> a = np.arange(1,11)
        >>> for i in range(0,9):
        ...     a[i+1] += a[i]
        >>> a
        array([1 3 6 10 15 21 28 36 45 55])

The same can be written in Bohrium as::

        >>> def loop_body(a):
        ...    i = get_iterator()
        ...    a[i+1] += a[i]
        >>> a = bh.arange(1,11)
        >>> bh.do_while(loop_body, 9, a)
        >>> a
        array([1 3 6 10 15 21 28 36 45 55])

Example 2. Increasing every second element by one, starting at both ends, in the same loop. As it can be seen: `i` is increased by 2, while `j` is descreased by 2 for each iteration::

        >>> def loop_body(a):
        ...   i = get_iterator(1)
        ...   a[2*i] += a[2*(i-1)]
        ...   j = i+1
        ...   a[1-2*j] += a[1-2*(j-1)]
        >>> a = bh.ones(10)
        >>> bh.for_loop(loop_body, 4, a)
        >>> a
        array([1 5 2 4 3 3 4 2 5 1])

Nested loops is also available in :func:`bohrium.loop.do_while` by using grids. A grid is a set of iterators that depend on each other, just as with nested loops. A grid can have arbitrary size and is available via. the function :func:`bohrium.loop.get_grid()`, which is only usable within a :func:`bohrium.loop.do_while` loop body. The function takes an amount of integers as parameters, corresponding to the range of the loops (from outer to inner). It returns the same amount of iterators, which functions as a grid. An example of this can be seen in Example 3 below.
Example 3. Creating a range in an array with multiple dimensions. In Numpy it can be written as::

        >>> a = bh.zeros((3,3))
        >>> counter = bh.zeros(1)
        >>> for i in range(3):
        ...    for j in range(3):
        ...        counter += 1
        ...        a[i,j] += counter
        >>> a
        [[1. 2. 3.]
         [4. 5. 6.]
         [7. 8. 9.]]

The same can done within a ``do_while`` loop by using a grid::

        >>> def kernel(a, counter):
        ...    i, j = get_grid(3,3)
        ...    counter += 1
        ...    a[i,j] += counter
        >>> a = bh.zeros((3,3))
        >>> counter = bh.zeros(1)
        >>> bh.do_while(kernel, 3*3, a, counter)
        >>> a
        [[1. 2. 3.]
         [4. 5. 6.]
         [7. 8. 9.]]


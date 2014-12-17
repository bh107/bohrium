C library
-----------

The C interface introduces two array concepts:

    * A base array that has a `rank` (number of dimensions) and `shape` (array of dimension sizes). The memory of the base array is always a single contiguous block of memory.
    * A view array that, beside a `rank` and a `shape`, has a `start` (start offset in number of elements) and a `stride` (array of dimension strides in number of elements). The view array refers to a (sub)set of a underlying base array where `start` is the offset into the base array and `stride` is number of elements to skip in order to iterate one step in a given dimension.


API
~~~

The C interface consists of a broad range of functions -- in the following, we describe some of the important ones.


Create a new empty array with `rank` number of dimensions and with the shape `shape` and returns a handler/pointer to a `complete` view of this new array:

.. code-block:: c

    bh_multi_array_{TYPE}_p bh_multi_array_{TYPE}_new_empty(uint64_t rank, const int64_t* shape);

Get pointer/handle to the base of a view:

.. code-block:: c

    bh_base_p bh_multi_array_{TYPE}_get_base(const bh_multi_array_{TYPE}_p self);


Destroy the base array and the associated memory:

.. code-block:: c

    void bh_multi_array_{TYPE}_destroy_base(bh_base_p base);


Destroy the view and base array (but not the associated memory):

.. code-block:: c

    void bh_multi_array_{TYPE}_discard(const bh_multi_array_{TYPE}_p self);

Some meta-data access functions:

.. code-block:: c

    // Gets the number of elements in the array
    uint64_t bh_multi_array_{TYPE}_get_length(bh_multi_array_{TYPE}_p self);

    // Gets the number of dimensions in the array
    uint64_t bh_multi_array_{TYPE}_get_rank(bh_multi_array_{TYPE}_p self);

    // Gets the number of elements in the dimension
    uint64_t bh_multi_array_{TYPE}_get_dimension_size(bh_multi_array_{TYPE}_p self, const int64_t dimension);

Before accesses the memory of an array, one has to synchronize the array:

.. code-block:: c

    void bh_multi_array_{TYPE}_sync(const bh_multi_array_{TYPE}_p self);

Access the memory of an array (remember to synchronize):

.. code-block:: c

    bh_{TYPE}* bh_multi_array_{TYPE}_get_base_data(bh_base_p base);

Some of the element-wise operations:

.. code-block:: c

    //Addition
    void bh_multi_array_{TYPE}_add(bh_multi_array_{TYPE}_p out, const bh_multi_array_{TYPE}_p lhs, const bh_multi_array_{TYPE}_p rhs);

    //Multiply
    void bh_multi_array_{TYPE}_multiply(bh_multi_array_{TYPE}_p out, const bh_multi_array_{TYPE}_p lhs, const bh_multi_array_{TYPE}_p rhs);

    //Addition: scalar + array
    void bh_multi_array_{TYPE}_add_scalar_lhs(bh_multi_array_{TYPE}_p out, bh_{TYPE} lhs, const bh_multi_array_{TYPE}_p rhs);

Some of the reduction and accumulate (aka scan) functions where `axis` is the dimension to reduce/accumulate over:

.. code-block:: c

    //Sum
    void bh_multi_array_{TYPE}_add_reduce(bh_multi_array_{TYPE}_p out, const bh_multi_array_{TYPE}_p in, bh_int64 axis);

    //Prefix sum
    void bh_multi_array_{TYPE}_add_accumulate(bh_multi_array_{TYPE}_p out, const bh_multi_array_{TYPE}_p in, bh_int64 axis);




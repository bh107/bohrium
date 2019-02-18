C++ library
===========

The C++ interface of Bohrium is similar to NumPy but is still very basic.


Indexing / Slicing
~~~~~~~~~~~~~~~~~~

Bohrium C++ only support single index indexing:

.. code-block:: cpp

    // Create a new empty array (4 by 5)
    bhxx::BhArray<double> A = bhxx::empty<double>({4, 5});
    // Create view of the third row of A
    bhxx::BhArray<double> B = A[2];

If you need more flexible slicing, you can set the shape and stride manually:

.. code-block:: cpp

    // Create a new array (4 by 5) of ones
    bhxx::BhArray<double> A = bhxx::ones<double>({4, 5});
    // Create view of the complete A.
    bhxx::BhArray<double> B = A;
    // B is now a 2 by 5 view with a step of two in the first dimension.
    // In NumPy, this corresponds to: `B = A[::2, :]`
    B.setShapeAndStride({2, 5}, {10, 1});


Code Snippets
~~~~~~~~~~~~~

You can find some examples in the `source tree <https://github.com/bh107/bohrium/tree/master/bridge/cxx/examples>`_ and some code snippets here:

.. code-block:: cpp

    #include<bhxx/bhxx.hpp>

    /** Return a new empty array */
    bhxx::BhArray<double> A = bhxx::empty<double>({4, 5});

    /** Return the rank (number of dimensions) of the array */
    int rank = A.rank();

    /** Return the offset of the array */
    uint64_t offset = A.offset();

    /** Return the shape of the array */
    Shape shape = A.shape();

    /** Return the stride of the array */
    Stride stride = A.stride();

    /** Return the total number of elements of the array */
    uint64_t size = A.size();

    /** Return a pointer to the base of the array */
    std::shared_ptr<BhBase> base = A.base();

    /** Return whether the view is contiguous and row-major */
    bool is_contig = A.isContiguous();

    /** Return a new copy of the array */
    bhxx::BhArray<double> copy = A.copy();

    /** Return a copy of the array as a standard vector */
    std::vector<double> vec = A.vec();

    /** Print the content of A */
    std::cout << A << "\n";

    // Return a new transposed view
    bhxx::BhArray<double> A_T = A.transpose();

    // Return a new reshaped view (the array must be contiguous)
    bhxx::BhArray<double> A_reshaped = A.reshape(Shape shape);

    /** Return a new view with a "new axis" inserted.
     *
     *  The "new axis" is inserted just before `axis`.
     *  If negative, the count is backwards
     */
    bhxx::BhArray<double> A_new_axis = A.newAxis(1);

    // Return a new empty array
    auto A = bhxx::empty<float>({3,4});

    // Return a new empty array that has the same shape as `ary`
    auto B = bhxx::empty_like<float>(A);

    // Return a new array filled with zeros
    auto A = bhxx::zeros<float>({3,4});

    // Return evenly spaced values within a given interval.
    auto A = bhxx::arange(1, 3, 2); // start, stop, step
    auto A = bhxx::arange(1, 3); // start, stop, step=1
    auto A = bhxx::arange(3); // start=0, stop, step=1

    // Random array, interval [0.0, 1.0)
    auto A = bhxx::random.randn<double>({3, 4});

    // Element-wise `static_cast`.
    bhxx::BhArray<int> B = bhxx::cast<int>(A);

    // Alias, A and B points to the same underlying data.
    bhxx::empty<float> A = bhxx::empty<float>({3,4});
    bhxx::empty<float> B = A;

    // a is an alias
    void add_inplace(bhxx::BhArray<double> a,
                     bhxx::BhArray<double> b) {
        a += b;
    }
    add_inplace(A, B);

    // Create the data of A into a new array B.
    bhxx::empty<float> A = bhxx::empty<float>({3,4});
    bhxx::empty<float> B = A.copy();

    // Copy the data of B into the existing array A.
    A = B;

    // Copying and converting the data of A into C.
    bhxx::empty<double> C = bhxx::cast<double>(A);

    // Alias, A and B points to the same underlying data.
    bhxx::empty<float> A = bhxx::empty<float>({3,4});
    bhxx::empty<float> B = bhxx::empty<float>({4});
    B.reset(A);

    // Evaluation triggers:
    bhxx::flush();
    std::cout << A << "\n";
    A.vec();
    A.data();

    // Operator overloads
    A + B - C * E / G;

    // Standard functions
    bhxx::sin(A) + bhxx::cos(B) + bhxx::sqrt(C) + ...

    // Reductions (sum, product, maximum, etc.)
    bhxx::add_reduce(A, 0); // Sum of axis 0
    bhxx::multiply_reduce(B, 1); // Product of axis 1
    bhxx::maximum_reduce(C, 2); // Maximum of axis 2


The API
~~~~~~~

The following is the complete API as defined in the `header file <https://github.com/bh107/bohrium/tree/master/bridge/cxx/include/bhxx>`_:

.. doxygenindex::



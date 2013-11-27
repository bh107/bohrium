C++ library
-----------

Usage
~~~~~

C++ is notorious for its support of multiple programming paradigms. It is thus possible for a DSEL in C++ to be object-oriented, function-oriented, or a mixture of both orientations. A functional approach integrates well with operator overloads as it very closely mimic the syntax of mathematical expressions such as X=sin(Y)+Z. An object-oriented style does also hold its merit for mathematical expressions on variables such as: X^{t} which can be represented as X.transpose().

The Bohrium C++ library provides a functional paradigm as a means to keep the notation consistent and within a single paradigm close to the domain of mathematics. The following goes through the most common operations of the library and describes the notation by example.

.. code-block:: cpp

  // Declaration and initialization of variables
  multi_array<float> x, y, z;
  multi_array<int> q;

Declaring and defining variables are separate operations. The declaration as shown above is only concerned with providing a name and type of the array. The initialization defines the shape along with the actual data. In the example below are two vectors initialized with three ones and three pseudo-random numbers.

.. code-block:: cpp

 // Definition and initialization of variables
  x = random<float>(3);  // x = [0.225, 0.456, 0.965]
  y = ones<float>(3);    // y = [1.0, 1.0, 1.0]

Operands can, once declared and defined, be used as input for operations such as element-wise addition or reduction. Operands can be, once declared but not defined, used to store the result of an operation. They will then inherit the shape based on the result of the operation when assigned.

.. code-block:: cpp

  // Element-wise operations
  z = x + y;  // z = [1.225, 1.456, 1.965]
  // Reduction
  z = sum(z); // z = [4.645]

Operands refer the result of an operation or another variable. Directly assigning an variable to another will create a view or alias of the other variable. Given two variables x  and y , where y  is an alias of x  the effect of an alias is that any operation on y  will also affect x  and vice versa as illustrated in the example below.

.. code-block:: cpp

  // Aliasing
  y = x;              // y is an alias of x 
  y += 1;
  cout << x << endl;  // [1.225, 1.456, 1.965]

In case an actual copy of an variable is needed the user has to explicitly request a copy. Copies also occur implicitly when variables are type-cast. Both of these situations are illustrated below.

.. code-block:: cpp

  // Explicit copy elements of variables
  z = copy(x);       // z = [1.225, 1.456, 1.965]
  // Typecasting, copies implicitly
  q = to<int>(x);   // q = [1, 1, 2]

The definition/initialization assigns the shape of an variable. It can be changed at a later point in time as long as the number of elements remain the same. The code below provide a couple of shape transformation examples.

.. code-block:: cpp

  multi_array<float> u;
  u = random<float>(9);
  ...
  u = reshape(u, 3, 3); // Turn vector into a 3x3 matrix
  u = transpose(u);     // Transpose the matrix
  ...
  u = reshape(u, 9);    // Turn 3x3 matrix into a vector

We have so far covered how to describe alias and explicit copies. This leaves the notation for updating an variable. The code below show how to update either a part of or the entire variable.

.. code-block:: cpp

  y(x);             // Update every element
  y[_(0,-1,2)] = 2; // Update every second element

The update of the every second element in the example above introduces the slicing notation. This notation is the most brittle from a productivity perspective compared to the notations provided by languages such as Matlab, R, Python and Cilk Plus. However, it is close to as good as it gets when using a library-based approach.

.. code-block:: cpp

  y[_(0,-1,2)]  // Every second element
  y[_(0,-1,1)]  // All elements
  y[_(2,-1,1)]  // All but the first two
  y[_(2,-2,1)]  // All but the last two
  y[_(1,-2,1)]  // Every second but the first and the last

Further examples of the notation, as well as examples of applications such as Black-Scholes, Jacobi Stencil, and Monte Carlo Pi, can be inspected in the \lstinline!benchmark/cpp/src/! directory of the Bohrium source-code repository[approaches:bohrium:repository].

The DSEL supports basic functionality for legacy support with C++ in the form of the iterator-interface for element-wise traversal. Overload of the shift-operator provides a convenient means of outputting the contents of the array.

.. code-block:: cpp

  for(multi_array<float>::iterator it=y.begin(); it != y.end(); ++it) {
    printf("%d", *it);
  }
  ...
  cout << y << endl;

The use of the iterator is highly discouraged as it forces the variable to synchronize its memory with the C++ memory space. Each element needs to be exposed and printed to screen in the above example. The iterator forces memory, which could be distributed out on GPU device memory or distributed in a cluster, to be copied back into main-memory for the application to access it. The iterator should for this reason only be used at the end of an application when results from computations need to be reported back to the user of the application.

Example and Makefile
~~~~~~~~~~~~~~~~~~~~

Simplest example of compiling an application using the Bohrium C++ library.

.. code-block:: cpp

  #include <iostream>
  #include "bh/bh.hpp"

  using namespace std;
  using namespace bh;

  int main()
  {
      multi_array<double> x;
      x = ones<double>(3,3);
      x = x + x;

      cout << "Hello Twos!" << x << endl;

      return 0;
  }

A basic Makefile::

  ROOT=../..
  HEADER=$(ROOT)/include/*
  CPPB_INCLUDE=$(ROOT)/bridge/cpp

  CXX=g++
  EXTRAS+=
  CXXFLAGS=-Wall -Wextra -pedantic -g -O2 -std=c++0x $(EXTRAS)

  all: hello

  hello: hello.cpp $(HEADER)
    $(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++


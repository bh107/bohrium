Python/NumPy
============

Three Python packages of Bohrium exist:

    - **bohrium**: is a package that integrate into NumPy and accelerate NumPy operations seamlessly. Everything is completely automatic, which is great when it works but it also makes it hard to know why code does perform as expected.
    - **bh107**: is a package that provide a similar interface and similar semantic as NumPy but everything is explicit. However, it is very easy to convert a **bh107** array into a NumPy array without any data copying.
    - **bohrium_api**: as the name suggest, this packages implements the core Bohrium API, which **bohrium** and *bh107** uses. It is not targeting the end-user.

.. toctree::
   :maxdepth: 2

   bohrium/index.rst
   bh107/index.rst


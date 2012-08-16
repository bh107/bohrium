Mac OSX
-----

The following explains how to get going on Mac OSX. cphVB and NumPy is not compatible with the Python interpreter included with OSX. We recommend that you use Python version 2.7 from the `MacPorts <http://www.macports.org>`_ project. Furthermore, MacPorts have all packages that are needed to compile and install cphVB.

You need to install the `Xcode Developer Tools <https://developer.apple.com/technologies/tools/>`_ from Apple and the following packages from MacPorts::
  
 sudo port install python27

Download and extract the source code::
  
  wget http://cphvb.org/cphvb-v0.1.tgz
  tar -xzf cphvb-v0.1.tgz

When building and install cphVB we need to specify the newly installed Python interpreter. In this case we use Python version 2.7::
  
  cd cphvb-v0.1
  make CPHVB_PYTHON=python2.7
  make install CPHVB_PYTHON=python2.7

.. note:: The installation will prompt you for the installation path. 
          The default path is ``/opt/cphvb`` which requires root permissions. Hence, if you do not have root access use a installation path to inside your home directory.

Finally, you need to set the ``PYTHONPATH`` and the ``DYLD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed cphVB Python module. This will also make sure that Python uses the NumPy module included in cphVB::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=/opt/cphvb/lib/python2.7/site-packages:$PYTHONPATH

The ``DYLD_LIBRARY_PATH`` should include the path to the installation directory::

  export DYLD_LIBRARY_PATH=<install dir>:$DYLD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=/opt/cphvb:$DYLD_LIBRARY_PATH
  
Now the basic installation should work. Try running the NumPy test suite::

  python2.7 test/numpy/numpytest.py

And you should see a result similar to this::

    *** Testing the equivalency of cphVB-NumPy and NumPy ***
    Testing test_array_create.py/array_create/zeros
    Testing test_sor.py/sor/sor
    Testing test_primitives.py/cphvb_opcodes/ufunc
    Testing test_primitives.py/numpy_ufunc/ufunc
    Testing test_reduce.py/reduce/reduce
    Testing test_benchmarks.py/gameoflife/gameoflife
    Testing test_benchmarks.py/jacobi/jacobi
    Testing test_benchmarks.py/jacobi_stencil/jacobi_stencil
    Testing test_benchmarks.py/shallow_water/shallow_water
    Testing test_matmul.py/matmul/dot
    Testing test_matmul.py/matmul/matmul
    Testing test_views.py/diagonal/diagonal
    Testing test_views.py/flatten/flatten
    ************************ Finish ************************




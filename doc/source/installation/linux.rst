Linux
-----

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.

You need to install all packages required to build NumPy::
  
  sudo apt-get build-dep python-numpy  

Download and extract the source code::
  
  wget http://cphvb.org/cphvb-v0.1.tgz
  tar -xzf cphvb-v0.1.tgz

Build and install::
  
  cd cphvb-v0.1
  make
  make install

.. note:: The installation will prompt you for the installation path. 
          The default path is ``/opt/cphvb`` which requires root permissions. Hence, if you do not have root access use a installation path to inside your home directory.

Finally, you need to set the ``PYTHONPATH`` and the ``LD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed cphVB Python module. This will also make sure that Python uses the NumPy module included in cphVB::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=/opt/cphvb/lib/python2.7/site-packages:$PYTHONPATH

The ``LD_LIBRARY_PATH`` should include the path to the installation directory::

  export LD_LIBRARY_PATH=<install dir>:$LD_LIBRARY_PATH
  #Example
  export LD_LIBRARY_PATH=/opt/cphvb:$LD_LIBRARY_PATH
  
Now the basic installation should work. Try running the NumPy test suite::

  python test/numpy/numpytest.py

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


OpenCL
~~~~~~

In order to support the OpenCL backend you might just need this::

  sudo apt-get install -y rpm alien libnuma1
  tar zxf intel_sdk_for_ocl_applications_2012_x64.tgz
  sudo apt-get install -y rpm alien libnuma1
  fakeroot alien --to-deb intel_ocl_sdk_2012_x64.rpm
  sudo dpkg -i intel-ocl-sdk_2.0-31361_amd64.deb
  sudo ln -s /usr/lib64/libOpenCL.so /usr/lib/libOpenCL.so
  sudo ldconfig


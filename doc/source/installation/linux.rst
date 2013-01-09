Linux
-----

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.

You need to install all packages required to build NumPy::
  
  sudo apt-get build-dep python-numpy  

And some additional packages::

  sudo apt-get install g++ python-dev python-pip python-cheetah python-sphinx doxygen libmpich2-dev
  sudo pip install breathe numpydoc

Download and extract the source code::
  
  wget https://bitbucket.org/cphvb/cphvb/downloads/cphvb-v0.1.tgz
  tar -xzf cphvb-v0.1.tgz

Build and install::
  
  cd cphvb-v0.1
  make
  make install

.. note:: The installation will prompt you for the installation path. 
          The default path is ``/opt/cphvb`` which requires root permissions. Hence, if you do not have root access use a installation path to inside your home directory.

.. note:: To compile to a custom Python (with valgrind debug support for example), set the make variable, CPHVB_PYTHON, naming the binary of your custom compiled Python.

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


OpenCL / GPU Engine
~~~~~~~~~~~~~~~~~~~

The GPU vector engine requires OpenCL compatible hardware as well as functioning drivers.
Configuring your GPU with you operating system is out of scope of this documentation.

Assuming that your GPU-hardware is functioning correctly you need to install an OpenCL SDK and some additional packages.

**Packages**::

  sudo apt-get install -y rpm alien libnuma1

**SDK for OpenCL**

Go to http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/ and download *Intel SDK for OpenCL 2012 -- Linux*.

The download-button is in the upper right corner next to select-box with the text *Select version...*.

The download area is hard to spot, so take a look at the red arrow on the picture below:

.. image:: opencl_download.png
   :scale: 50 %
   :alt: Download location.

Once downloaded, install the SDK with the following commands::

  tar zxf intel_sdk_for_ocl_applications_2012_x64.tgz
  fakeroot alien --to-deb intel_ocl_sdk_2012_x64.rpm
  sudo dpkg -i intel-ocl-sdk_2.0-31361_amd64.deb
  sudo ln -s /usr/lib64/libOpenCL.so /usr/lib/libOpenCL.so
  sudo ldconfig

You should now have everything you need to utilize the GPU engine.

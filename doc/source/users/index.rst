Usage Guide
==================

...

Download / Install / Configure
------------------------------

Linux
~~~~~

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.

  sudo apt-get install python-dev gfortran

Set the environment variables $LD_LIRBARY and $PYTHONPATH.

Get the source, compile and install it::

  wget ...
  sudo mkdir /opt/cphvb
  sudo chown jane:jane
  ./build.py install


OpenCL
~~~~~~

You might just need this::

  sudo apt-get install -y rpm alien libnuma1
  tar zxf intel_sdk_for_ocl_applications_2012_x64.tgz
  sudo apt-get install -y rpm alien libnuma1
  fakeroot alien --to-deb intel_ocl_sdk_2012_x64.rpm
  sudo dpkg -i intel-ocl-sdk_2.0-31361_amd64.deb
  sudo ln -s /usr/lib64/libOpenCL.so /usr/lib/libOpenCL.so
  sudo ldconfig

...

MacOSX
~~~~~~

...

Windows
~~~~~~~

...

Configure
---------

Create a configuration file. See config-example... configuration search-path::

  <CWD>/config.ini
  <PATH>/config.ini
  ~/.cphvb/config.ini
  /opt/cphvb/config.ini

Sample configuration file cphvb/sample-config.ini::

  ...

Environment variables::

  PYTHONPATH="$PYTHONPATH:/opt/cphvb/lib/python2.7/site-packages"
  LD_LIBRARY_PATH="$PATH:/opt/cphvb"

Examples
--------

...

Advanced Features
-----------------

...


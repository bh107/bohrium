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
  ./build install


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


Installation using Spack
========================

This guide will install Bohrium using the Spack package manager.

Why use Spack?
--------------
`Spack <https://spack.io/>`_ is a package management tool tailored
specifically for supercomputers with a rather dated software stack.
It allows to install and maintain packages, starting only from
very `few dependencies <https://spack.readthedocs.io/en/latest/getting_started.html>`_:
Pretty much just python2.6, git, curl and some c++ compiler are all
that's needed for the bootstrap.

Needless to say that the request for installing a particular package
automatically yields the installation of all dependencies with
exactly the right version and configurations. If this causes
multiple versions/configurations of the same package to be required,
this is no problem and gets resolved automatically, too.
As a bonus on top, using an installed package later is super easy
as well due to an automatic generation of module files,
which set the required environment up.

Installation overview
---------------------

.. highlight:: ruby

First step is to clone and setup Spack::

  export SPACK_ROOT="$PWD/spack"
  git clone https://github.com/llnl/spack.git
  . $SPACK_ROOT/share/spack/setup-env.sh

Afterwards the installation of Bohrium is instructed::

  spack install bohrium

This step will take a while, since Spack will download the sources of all dependencies,
unpack, configure and compile them. But since everything happens in the right order
automatically, you could easily do this over night.

That's it. If you want to use Bohrium, setup up Spack as above,
then load the required modules::

  spack module loads -r bohrium > /tmp/bohrium.modules
  . /tmp/bohrium.modules

and you are ready to go as the shell environment now contains
all required variables (`LD_LIBRARY_PATH`, `PATH`, `CPATH`, `PYTHONPATH`, ...)
to get going.

If you get some errors about the command `module` not being found, you need
to install the Spack package `environment-modules` beforehand. Again,
just a plain::

  spack install environment-modules

is enough to achieve this.

Tuning the installation procedure
---------------------------------

Spack offers countless ways to influence how things are installed and
what is installed. See the `Documentation <https://spack.readthedocs.io>`_
and especially the
`Getting Started <https://spack.readthedocs.io/en/latest/getting_started.html>`_
section for a good overview.

Most importantly the so-called `spec` allows to specify features or requirements
with respect to versions and dependencies,
that should be enabled or disabled when building the package.
For example::

  spec install bohrium~cuda~opencl

.. highlight:: python

Will install Bohrium *without* CUDA or OpenCL support, which has a dramatic impact
on the install time due to the reduced amount of dependencies to be installed.
On the other hand::

  spec install bohrium@develop

will install specifically the development version of Bohrium.
This the current `HEAD` of the `master` branch in the github repository.
One may also influence the versions of the dependencies by themselves.
For example::

  spec install bohrium+python^python@3:

will specifically compile Bohrium with a python version larger than 3.

The current list of features the Bohrium package has to offer can be listed
by the command::

  spack info bohrium

and the list of dependencies which will be installed by a particlar `spec`
can be easily reviewed by something like::

  spack spec bohrium@develop~cuda~opencl

Installation as non-priviliged user on a system with a dated software-stack 
===========================================================================

Some clusters have quite dated software stacks, this documents how to install basically everything needed to bootstrap something never. This will install:

 * gcc 4.8.2
 * python 2.7.10
 * cmake 3.3.0
 * boost 1.58
 * swig 3.0.6
 * pcre 8.37
 * htop 1.0.3 (optional)
 * bash 4.3 (optional)
 * Python packages via pip: `cheetah`, `cython`, and `numpy`
 * Benchpress and Bohrium via git

Create some folder for all prerequisites::

  mkdir $HOME/preqs

Set environment vars, you probably want to persist it (.profile, .bashrc, or .bash_aliases)::

  export CPLUS_INCLUDE_PATH=$HOME/aux/boost-1.58.0/include:$CPLUS_INCLUDE_PATH
  export CPLUS_INCLUDE_PATH=$HOME/aux/pcre-8.37/include:$CPLUS_INCLUDE_PATH
  export CPLUS_INCLUDE_PATH=$HOME/aux/gcc-4.8.2/include:$CPLUS_INCLUDE_PATH
  export LD_LIBRARY_PATH=$HOME/aux/boost-1.58.0/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/aux/pcre-8.37/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/aux/gcc-4.8.2/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/aux/gcc-4.8.2/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
  export PATH=$HOME/aux/htop-1.0.3/bin:$PATH
  export PATH=$HOME/aux/bash-4.3/bin:$PATH
  export PATH=$HOME/aux/cmake-3.3.0/bin:$PATH
  export PATH=$HOME/aux/python-2.7.10/bin:$PATH
  export PATH=$HOME/aux/gcc-4.8.2/bin:$PATH
  export PATH=$HOME/aux/benchpress/bin:$PATH
  export PATH=$HOME/aux/pcre-8.37/bin:$PATH
  export PATH=$HOME/aux/swig-3.0.6/bin:$PATH
  export PYTHONPATH=$HOME/aux/benchpress/module:$PYTHONPATH

Be warned, this is a fairly time-consuming task. Expect 3-4 hours.
The most time consuming are compiling `gcc` and `boost`.

.. note:: 
  
  The order that you perform the following is quite important,
  you want to get a recent `gcc` before compiling anything else since anything else
  would othervise be compiled with an older `gcc`.

gcc 4.8.2
---------

Start by installing `gcc 4.8` this probably takes a couple of hours::

  cd $HOME/preqs

  # Download and extract gcc itself
  wget ftp://ftp.gnu.org/gnu/gcc/gcc-4.8.2/gcc-4.8.2.tar.gz
  tar xzf gcc-4.8.2.tar.gz

  # Download and unpack dependencies needed by gcc
  wget http://www.multiprecision.org/mpc/download/mpc-1.0.1.tar.gz
  tar xfz mpc-1.0.1.tar.gz
  mv mpc-1.0.1 gcc-4.8.2/mpc

  wget http://www.mpfr.org/mpfr-current/mpfr-3.1.3.tar.gz
  tar xzf mpfr-3.1.3.tar.gz
  mv mpfr-3.1.3 gcc-4.8.2/mpfr

  wget https://gmplib.org/download/gmp/gmp-5.1.3.tar.bz2
  tar -jxf gmp-5.1.3.tar.bz2
  mv gmp-5.1.3 gcc-4.8.2/gmp

  wget ftp://ftp.irisa.fr/pub/mirrors/gcc.gnu.org/gcc/infrastructure/isl-0.11.1.tar.bz2
  tar -jxf isl-0.11.1.tar.bz2
  mv isl-0.11.1 gcc-4.8.2/isl

  wget ftp://ftp.irisa.fr/pub/mirrors/gcc.gnu.org/gcc/infrastructure/cloog-0.18.0.tar.gz
  tar xfz cloog-0.18.0.tar.gz
  mv cloog-0.18.0 gcc-4.8.2/cloog

  mkdir $HOME/aux/gcc-4.8.2
  cd gcc-4.8.2
  ./configure --prefix=$HOME/aux/gcc-4.8.2 --enable-languages=c,c++ --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --enable-gnu-unique-object --disable-libmudflap --enable-plugin --enable-multiarch --with-tune=generic --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu                                                                                 
  make
  make -k check
  make install

This is the most time-consuming so go do something else.

And quite importantly make sure to link `gcc` to `cc`::

  cd $HOME/aux/gcc-4.8.2/bin
  ln -s gcc cc

Once it is done then verify that it gets called when invoking `gcc` and `cc`::

  gcc -v
  cc -V

If it does not then check your `$PATH`.

python 2.7.10
-------------

Then install `python`::

  cd $HOME/preqs
  wget https://www.python.org/ftp/python/2.7.10/Python-2.7.10.tgz
  tar xzf Python-2.7.10.tgz
  cd Python-2.7.10
  mkdir -p aux/python2.7
  ./configure --prefix=$HOME/aux/python2.7
  make install

And check that it called when invoking `python`::

  python -V

If it does not then check your `$PATH`.

Then bootstrap `pip`::

  cd $HOME/preqs
  wget https://bootstrap.pypa.io/get-pip.py
  python get-pip.py

We will need `pip` later for installing Python packages.

cmake 3.3.0
-----------

Continue with `cmake`::

  mkdir -p $HOME/aux/cmake
  cd $HOME/aux/cmake
  wget http://www.cmake.org/files/v3.3/cmake-3.3.0-rc4-Linux-x86_64.sh
  chmod +x cmake-3.3.0-rc4-Linux-x86_64.sh
  ./cmake-3.3.0-rc4-Linux-x86_64.sh

Just follow the wizard.

boost 1.58.0
------------

Then install `boost`::

  cd $HOME/install
  wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download -O boost.tar.gz
  tar xzf boost.tar.gz
  cd boost*
  mkdir $HOME/aux/boost
  ./bootstrap.sh --prefix=$HOME/aux/boost
  ./b2 --with-serialization --with-filesystem --with-system --with-thread install

This is the second most time-consuming compile you have to do.

swig 3.0.6
----------

Install `swig` and its dependency `pcre`.

Install `pcre 8.37`::

  cd $HOME/preqs
  wget ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.37.tar.gz
  tar xzf pcre-8.37.tar.gz
  cd pcre-8.37
  ./configure --prefix=$HOME/aux/pcre-8.37 --enable-unicode-properties --enable-pcre16 --enable-pcre32 --enable-pcregrep-libz --enable-pcregrep-libbz2 --enable-pcretest-libreadline --disable-static
  make
  make install

And then on to `swig` itself::

  cd $HOME/preqs
  wget http://prdownloads.sourceforge.net/swig/swig-3.0.6.tar.gz
  tar xfz swig-3.0.6.tar.gz
  cd swig-3.0.6
  ./configure --prefix=$HOME/aux/swig-3.0.6
  make
  make install

Bohrium works with even some of the oldest swig versions but if it is not available then go ahead and install it.

htop (optional)
---------------

I just like this `htop` over `top` but it is completely optional::

  cd $HOME/preqs
  wget http://hisham.hm/htop/releases/1.0.3/htop-1.0.3.tar.gz
  tar xzf htop-1.0.3.tar.gz
  ./configure --prefix=$HOME/aux/htop-1.0.3
  make
  make install

It is just such a nice convenience.

bash (optional)
------------------------

In case even your shell is broken then go for installing bash::

  cd $HOME/preqs
  wget http://git.savannah.gnu.org/cgit/bash.git/snapshot/bash-master.tar.gz
  tar xzf bash-master.tar.gz
  cd bash-master
  ./configure --prefix=$HOME/aux/bash-4.3
  make
  make install

Python Packages
---------------

These should now be installable via `pip`::

  pip install cheetah cython numpy

benchpress
----------

We need this to run testing against benchmarks and to run benchmarks from the benchpress repos::

  cd $HOME/aux
  git clone https://github.com/bh107/benchpress.git

Verify that you can invoke benchpress::

  bp-info --all

bohrium
-------

And now we can get on with installing bohrium::

  cd $HOME/aux
  git clone https://github.com/bh107/bohrium.git
  mkdir b
  cd b
  cmake ../ -DBRIDGE_CIL=OFF -DBRIDGE_NUMCIL=OFF -DVEM_CLUSTER=OFF -DVEM_PROXY=OFF -DEXT_VISUALIZER=OFF -DVE_GPU=OFF -DTEST_CIL=OFF -DBENCHMARK_CIL=OFF -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=$HOME/aux/boost-1.58.0 -DBoost_INCLUDE_DIRS=$HOME/aux/boost-1.58.0/include -DBoost_LIBRARY_DIRS=$HOME/aux/boost-1.58.0/lib -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON
  make
  make install
  ln -s $HOME/aux/bohrium-master $HOME/bohrium

Now run numpytest to check that it is operational::

  python $HOME/bohrium/test/python/numpytest.py

That only took a day... great.

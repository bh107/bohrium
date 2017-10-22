Installation as non-privileged user on a system with a dated software-stack
===========================================================================

In case you want to experiment with Bohrium in a restricted environment such as on a native mode on a Xeon Phi or on a cluster
with dated software stacks, this documents how to install basically everything needed to bootstrap something newer.

This will install:

 * gcc 4.9.3
 * htop 1.0.3 (optional)
 * bash 4.3 (optional)
 * binutils 2.25
 * python 2.7.10
 * cmake 3.3.0
 * boost 1.58
 * clang 3.5.0 with OpenMP (optional)
 * swig 3.0.6
 * pcre 8.37
 * Python packages via pip: `cython`, and `numpy`
 * Benchpress via git
 * Bohrium via git

Create some folder for all prerequisites, it will usually be fastest to use some local storage instead of a networked file system::

  mkdir /tmp/preqs
  mkdir $HOME/tools

Set environment vars (you probably want to make them persistent e.g. in your .profile, .bashrc or .bash_aliases)::

  export CPLUS_INCLUDE_PATH=$HOME/tools/boost-1.58.0/include:$CPLUS_INCLUDE_PATH
  export CPLUS_INCLUDE_PATH=$HOME/tools/pcre-8.37/include:$CPLUS_INCLUDE_PATH
  export CPLUS_INCLUDE_PATH=$HOME/tools/gcc-4.9.3/include:$CPLUS_INCLUDE_PATH
  export CPLUS_INCLUDE_PATH=$HOME/tools/binutils-2.25/include:$CPLUS_INCLUDE_PATH
  export LD_LIBRARY_PATH=$HOME/tools/boost-1.58.0/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/tools/pcre-8.37/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/tools/gcc-4.9.3/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/tools/gcc-4.9.3/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/tools/binutils-2.25/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
  export PATH=$HOME/tools/htop-1.0.3/bin:$PATH
  export PATH=$HOME/tools/cmake-3.3.0/bin:$PATH
  export PATH=$HOME/tools/python-2.7.10/bin:$PATH
  export PATH=$HOME/tools/gcc-4.9.3/bin:$PATH
  export PATH=$HOME/tools/benchpress/bin:$PATH
  export PATH=$HOME/tools/bash-4.3/bin:$PATH
  export PATH=$HOME/tools/pcre-8.37/bin:$PATH
  export PATH=$HOME/tools/swig-3.0.6/bin:$PATH
  export PATH=$HOME/tools/binutils-2.25/bin:$PATH
  export PYTHONPATH=$HOME/tools/benchpress/module:$PYTHONPATH

Be warned, this is a fairly time consuming task. Expect 3-4 hours.
The most time consuming steps are compiling `gcc` and `boost`.

.. note::

  The order that you perform the following is quite important,
  you want to get a recent `gcc` before compiling anything else since anything else
  would otherwise be compiled with an older `gcc`. Also installing binutils before
  gcc will probably result in compile-errors.

gcc 4.9.3
---------

Let's try with a newer gcc and a different preq-install::

  cd /tmp/preqs
  wget ftp://ftp.gnu.org/gnu/gcc/gcc-4.9.3/gcc-4.9.3.tar.gz
  tar xzf gcc-4.9.3.tar.gz
  cd gcc-4.9.3
  ./contrib/download_prerequisites
  ./configure --prefix=$HOME/tools/gcc-4.9.3 --enable-languages=c,c++ --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --enable-gnu-unique-object --disable-libmudflap --enable-plugin --enable-multiarch --with-tune=generic --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu
  make -j$(nproc)
  make install

This is the most time consuming, so you might just go do something else in the meantime.

And quite importantly make sure to link `gcc` to `cc`::

  cd $HOME/tools/gcc-4.9.3/bin
  ln -s gcc cc

Once it is done then verify that it gets called when invoking `gcc` and `cc`::

  gcc -v
  cc -V

If it does not then check your `$PATH`.

htop (optional)
---------------

I just like this `htop` over `top` but it is completely optional::

  cd /tmp/preqs
  wget http://hisham.hm/htop/releases/1.0.3/htop-1.0.3.tar.gz
  tar xzf htop-1.0.3.tar.gz
  cd htop-1.0.3
  ./configure --prefix=$HOME/tools/htop-1.0.3
  make -j$(nproc)
  make install

It is just such a nice convenience.

bash (optional)
------------------------

In case even your shell is broken then go for installing bash::

  cd /tmp/preqs
  wget http://git.savannah.gnu.org/cgit/bash.git/snapshot/bash-master.tar.gz
  tar xzf bash-master.tar.gz
  cd bash-master
  ./configure --prefix=$HOME/tools/bash-4.3
  make -j$(nproc)
  make install

mosh (optional)
---------------

  cd /tmp/preqs
  wget https://github.com/google/protobuf/archive/master.zip
  unzip master.zip
  cd master
  ./autogen.sh
  ./configure --prefix=$HOME/tools/protoc
  make
  make check
  make install

binutils 2.25
-------------

We need a newer assembler for avx::

  cd /tmp/preqs
  wget http://ftp.gnu.org/gnu/binutils/binutils-2.25.tar.gz
  tar xzf binutils-2.25.tar.gz
  cd binutils-2.25
  ./configure --prefix=$HOME/tools/binutils-2.25
  make -j$(nproc)
  make install

python 2.7.10
-------------

Then install `python`::

  cd /tmp/preqs
  wget https://www.python.org/ftp/python/2.7.10/Python-2.7.10.tgz
  tar xzf Python-2.7.10.tgz
  cd Python-2.7.10
  mkdir -p tools/python2.7
  ./configure --prefix=$HOME/tools/python2.7
  make install

And check that it's called when invoking `python`::

  python -V

If not then check your `$PATH`.

Then bootstrap `pip`::

  cd /tmp/preqs
  wget https://bootstrap.pypa.io/get-pip.py
  python get-pip.py

We will need `pip` later for installing Python packages.

cmake 3.3.0
-----------

Continue with `cmake`::

  mkdir $HOME/tools/cmake-3.3.0
  cd $HOME/tools/cmake-3.3.0
  wget http://www.cmake.org/files/v3.3/cmake-3.3.0-rc4-Linux-x86_64.sh
  chmod +x cmake-3.3.0-rc4-Linux-x86_64.sh
  ./cmake-3.3.0-rc4-Linux-x86_64.sh

Just follow the wizard.

Verify version::

  cmake --version

boost 1.58.0
------------

Then install `boost`::

  cd /tmp/preqs
  wget "http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download" -O boost.tar.gz
  tar xzf boost.tar.gz
  cd boost*
  ./bootstrap.sh --prefix=$HOME/tools/boost-1.58.0
  ./b2 --with-serialization --with-filesystem --with-system --with-thread install

This is the second most time consuming compile you have to do.

swig 3.0.5
----------

Install `swig` and its dependency `pcre`.

Install `pcre 8.37`::

  cd /tmp/preqs
  wget ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.37.tar.gz
  tar xzf pcre-8.37.tar.gz
  cd pcre-8.37
  ./configure --prefix=$HOME/tools/pcre-8.37 --enable-unicode-properties --enable-pcre16 --enable-pcre32 --enable-pcregrep-libz --enable-pcregrep-libbz2 --disable-static
  make -j$(nproc)
  make install

And then on to `swig` itself::

  cd /tmp/preqs
  wget http://prdownloads.sourceforge.net/swig/swig-3.0.5.tar.gz
  tar xfz swig-3.0.5.tar.gz
  cd swig-3.0.5
  ./configure --prefix=$HOME/tools/swig-3.0.5
  make -j$(nproc)
  make install

Bohrium works with even some of the oldest swig versions but if it is not available then go ahead and install it.

Python Packages
---------------

These should now be installable via `pip`::

  pip install cython 'numpy==1.8.2'

clang 3.5 with OMP
------------------

Without OpenMP clang is not of much use to Bohrium, so we grab the omp-port::

  cd /tmp/preqs
  git clone https://github.com/clang-omp/llvm
  git clone https://github.com/clang-omp/compiler-rt llvm/projects/compiler-rt
  git clone -b clang-omp https://github.com/clang-omp/clang llvm/tools/clang
  mkdir clang
  cd clang
  cmake ../llvm -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$HOME/tools/clang-3.5.0 -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  make install

Also build the Intel OpenMP runtime.

benchpress
----------

We need this to run testing against benchmarks and to run benchmarks from the benchpress repository::

  cd $HOME/tools
  git clone https://github.com/bh107/benchpress.git

Verify that you can invoke benchpress::

  bp-info --all

bohrium
-------

And now we can get on with installing bohrium::

  cd $HOME/tools
  git clone https://github.com/bh107/bohrium.git
  cd bohrium
  mkdir b
  cd b
  cmake ../ -DBRIDGE_CIL=OFF -DBRIDGE_NUMCIL=OFF -DVEM_CLUSTER=OFF -DVEM_PROXY=OFF -DEXT_VISUALIZER=OFF -DVE_GPU=OFF -DTEST_CIL=OFF -DBENCHMARK_CIL=OFF -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=$HOME/tools/boost-1.58.0 -DBoost_INCLUDE_DIRS=$HOME/tools/boost-1.58.0/include -DBoost_LIBRARY_DIRS=$HOME/tools/boost-1.58.0/lib -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON
  make -j$(nproc)
  make install
  ln -s $HOME/tools/bohrium-master $HOME/bohrium

Now run numpytest to check that it is operational::

  python $HOME/bohrium/test/python/numpytest.py

And you're done!

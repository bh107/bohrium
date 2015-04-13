Windows
-------

.. warning:: The newest version of Bohrium (v0.2) does not support Windows. Bohrium on Windows only supports the much older version 0.1. We are working on a new windows version but for now, we do not recommend using Bohrium on windows.

The follow explains how to get Bohrium installed on Windows. Note that only 64bit Windows is supported by the Bohrium binaries (the NumCIL binaries are 32bit and 64bit compatible).

.. The Bohrium package is distributed as a zip archive, that you can get from here:
    https://bitbucket.org/bohrium/bohrium/downloads/Bohrium-v0.1-win.zip

..  Simply extract the contents of the folder. If you run your program from the folder where the files reside it will work correctly.

.. You can either place the dll files in the folder of the project you are working on, or place them somewhere on your machine, and change your PATH environment variable to include this location. See the guide `How to change your path environment variable <http://www.computerhope.com/issues/ch000549.htm>`_.

..  If you want to place the files somewhere so multiple programs can use them, we recommend that you use "%PROGRAMFILES%\bohrium".
.. For an installation with shared libraries, you should edit the file config.ini and set all absolute paths to libraries. The config.ini file should then be placed in %PROGRAMFILES%\bohrium\config.ini.

You also need the `Microsoft Visual C++ 2010 Runtime <http://www.microsoft.com/en-us/download/details.aspx?id=14632>`_ installed.

Building from source
~~~~~~~~~~~~~~~~~~~

You need to install either `Microsoft Visual C++ 2010 <http://msdn.microsoft.com/en-us/library/vstudio/60k1461a(v=vs.100).aspx>`_ or `Microsoft Visual C++ Express 2010 <https://www.microsoft.com/visualstudio/eng/products/visual-studio-express-products>`_.

You also need `Python 2.7 <http://www.python.org/download/>`_ installed to run the build script.

You then need to download and extract the `source code <https://bitbucket.org/bohrium/bohrium/downloads/bohrium-v0.1.tgz>`_.

Once you have `Visual C <https://www.microsoft.com/visualstudio/eng/products/visual-studio-express-products>`_, `Python 2.7 <http://www.python.org/download/>`_ and the `source  <https://bitbucket.org/bohrium/bohrium/downloads/bohrium-v0.1.tgz>`_ go to a command prompt, and change to the folder where the source is installed::

   cd Downloads\bohrium

And set the environment to x64 and build it::

   SetEnv.cmd /Release /x64
   python build.py build

.. note:: The Bohrium NumPy module does not currently build under Windows, so you will get an error at the end.

You can then use the binaries from the locations, or build a package::

   cd misc
   make-win-release


If you had 7zip installed, you should now have a file called Bohrium.zip, otherwise you can access the files from the Bohrium-release folder.


Building latest version from source (does not work)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are notes of trying to get the latest Bohrium version running on Windwos 7 64-bit.
It did not succeed, first build-issue is the use "sys/mman.h" which is Linux specific.

Download and install the following prerequisites, in order:

 * Python 2.7 (https://www.python.org/downloads/)
 * CMake (http://www.cmake.org/)
 * Visual Studio (https://www.visualstudio.com/)
 * Visual C++ for Python 2.7 (http://aka.ms/vcpython27)
 * Boost (http://www.boost.org/)
 * SWIG (http://www.swig.org/download.html)
 * HWLOC (http://www.open-mpi.org/software/hwloc/v1.10/)
 * Cheetah
 * Cython
 * NumPy

Python, Cmake, Visual Studio and Visual C++ for Python 2.7 are no-fuss installation via .msi packages. Notes on Boost, SWIG, and HWLOC follow.

Boost, download and install boost to ``C:\boost``, it is around 200MB and consumes about 3GB once installed. After the installation-wizard has run, set the following env-var::

  SET BOOST_LIBRARYDIR=C:\boost\lib64-msvc-12.0

SWIG, unpack the archive and expend ``%PATH%`` to include it::

  SET PATH=%PATH%;C:\swig

HWLOC, currently Visual Studio does not have sufficient support for OpenMP so getting HWLOC installed is not relevant until the OpenMP issue is resolved.

Cheetah, Cython, NumPy, install these packages via pip::

  pip install cython cheetah numpy

With these things in place it is possible to start the Build of Bohrium.
However, as noted, it does not build due to various portability issues.

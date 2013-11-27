Windows
-------

The follow explains how to get Bohrium installed on Windows. Note that only 64bit Windows is supported by the Bohrium binaries (the NumCIL binaries are 32bit and 64bit compatible).

The Bohrium package is distributed as a zip archive, that you can get from here:
https://bitbucket.org/bohrium/bohrium/downloads/Bohrium-v0.1-win.zip

Simply extract the contents of the folder. If you run your program from the folder where the files reside it will work correctly.

You can either place the dll files in the folder of the project you are working on, or place them somewhere on your machine, and change your PATH environment variable to include this location. See the guide `How to change your path environment variable <http://www.computerhope.com/issues/ch000549.htm>`_.

If you want to place the files somewhere so multiple programs can use them, we recommend that you use "%PROGRAMFILES%\bohrium".
For an installation with shared libraries, you should edit the file config.ini and set all absolute paths to libraries. The config.ini file should then be placed in %PROGRAMFILES%\bohrium\config.ini.

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

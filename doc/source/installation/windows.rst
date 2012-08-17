Windows
-------

The follow explains how to get cphVB installed on Windows. Note that only 64bit Windows is supported by the cphVB binaries (the NumCIL binaries are 32bit and 64bit compatible).

The cphVB package is distributed as a zip archive, that you can get from here:
https://cphvb.org/download

Simply extract the contents of the folder. If you run your program from the folder where the files reside it will work correctly.

You can either place the dll files in the folder of the project you are working on, or place them somewhere on your machine, and change your PATH environment variable to include this location. See the guide `How to change your path environment variable <http://www.computerhope.com/issues/ch000549.htm>`_.

If you want to place the files somewhere so multiple programs can use them, we recommend that you use "%PROGRAMFILES%\cphvb".
For an installation with shared libraries, you should edit the file config.ini and set all absolute paths to libraries. The config.ini file should then be placed in %PROGRAMFILES%\cphvb\config.ini.

You also need the `Microsoft Visual C++ 2010 Runtime <http://www.microsoft.com/en-us/download/details.aspx?id=14632>`_ installed.


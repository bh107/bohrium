Welcome!
========

[![Build Status](https://travis-ci.org/bh107/bohrium.svg?branch=master)](https://travis-ci.org/bh107/bohrium)

Bohrium provides a runtime environment for efficiently executing vectorized applications using your favourite programming language Python/NumPy, C#, F# on Linux, Windows and MacOS.

Forget handcrafting CUDA/OpenCL, forget threading, mutexes, and locks, use Bohrium!

Features
--------
|           | Architecture Support                   || Frontends                                 |||||
|-----------|:---------------:|:---------------------:|:-------------:|:-------------:|:---:|:--:|:--:|
|           |  Multi-Core CPU | Many-Core GPU         | Python2/NumPy | Python3/NumPy | C++ | C# | F# |
| Linux     |  ✓              | ✓                     | ✓             | ✓             | ✓   | ✓  | ✓  |
| MacOS     |                 |                       | ✓             |               | ✓   | ✓  | ✓  |
| Windows   |                 |                       |               |               |     |    |    |

The documentation is available at www.bh107.org

Installation
------------
On Ubuntu use apt-get:
```
sudo add-apt-repository ppa:bohrium/nightly
sudo apt-get update
sudo apt-get install bohrium
```

On MacOS use [Homebrew](https://brew.sh):
```
brew install bh107/homebrew-bohrium/bohrium
```

Alternatively, build from source:
```
wget https://github.com/bh107/bohrium/archive/master.zip
unzip master.zip
cd bohrium-master
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<path to install directory>
make install
```

Find the full installation guide at: http://bohrium.readthedocs.io/installation/


User Guide
----------

In order to use Bohrium, simply run your Python/NumPy program using the command line argument `-m bohrium`:

```
python -m bohrium my_numpy_app.py
```

In which case, all instances of `import numpy` is converted to `import bohrium` seamlessly. If you need to access the real numpy module use `import numpy_force`.

If you have [Jupyter](http://jupyter.org/) installed, you can use the magic command `%%bohrium` to achieve the same results. 

For the full user guide, which include C, C++, and .NET languages, see: http://bohrium.readthedocs.io/users/index.html

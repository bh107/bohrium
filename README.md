Welcome!
========

[![Build Status](https://travis-ci.org/bh107/bohrium.svg?branch=master)](https://travis-ci.org/bh107/bohrium) [![Anaconda-Server Badge](https://anaconda.org/bohrium/bohrium/badges/installer/conda.svg)](https://conda.anaconda.org/bohrium) [![Gitter Chat](https://badges.gitter.im/bh107/gitter.png)](https://gitter.im/bh107/Lobby)
[![Documentation Status](https://readthedocs.org/projects/bohrium/badge/?version=latest)](http://bohrium.readthedocs.io/?badge=latest)

Bohrium provides a runtime environment for efficiently executing vectorized applications using your favourite programming language Python/NumPy, C#, F# on Linux, Windows and MacOS.

Forget handcrafting CUDA/OpenCL, forget threading, mutexes, and locks, use Bohrium!

Features
--------
|           | Architecture Support                   || Frontends                                 |||||
|-----------|:---------------:|:---------------------:|:-------------:|:-------------:|:---:|:--:|:--:|
|           |  Multi-Core CPU | Many-Core GPU         | Python2/NumPy | Python3/NumPy | C++ | C# | F# |
| Linux     |  ✓              | ✓                     | ✓             | ✓             | ✓   | ✓  | ✓  |
| MacOS     |                 | ✓                     | ✓             |               | ✓   | ✓  | ✓  |
| Windows   |                 |                       |               |               |     |    |    |

- **Lazy Evaluation**, Bohrium will lazy evaluate all Python/NumPy operations until it encounters a “Python Read” such a printing an array or having a if-statement testing the value of an array.
- **Views** Bohrium supports NumPy views fully thus operating on array slices does not involve data copying.
- **Loop Fusion**, Bohrium uses a [fusion algorithm](http://dl.acm.org/citation.cfm?id=2967945) that fuses (or merges) array operations into the same computation kernel that are then JIT-compiled and executed. However, Bohrium can only fuse operations that have some common sized dimension and no horizontal data conflicts.
- **Lazy CPU/GPU Communiction**, Bohrium only move data between the host and the GPU when the data is accessed directly by Python or a Python C-extension.

The documentation is available at www.bh107.org

Installation
------------
On Ubuntu use apt-get:
```
sudo add-apt-repository ppa:bohrium/nightly
sudo apt-get update
sudo apt-get install bohrium
sudo apt-get install bohrium-opencl # For GPU support
```

On Linux-64 use [Anaconda](https://www.continuum.io/downloads) (currently, no GPU support):
```
# Create a new environment 'bh' with the 'bohrium' package from the 'bohrium' channel:
conda create -n bh -c bohrium bohrium
# And source the new environment:
source activate bh

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

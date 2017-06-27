MacOS
-----

The following explains how to get going on MacOS. You need some extra packages to build Bohrium from source.

You need to install the `Xcode Developer Tools <https://developer.apple.com/xcode/>`_ package, which is found in the App Store.

Simple install with Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by `installing Homebrew as explained on their website <http://brew.sh/>`_ ::

  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Bohrium::

  python -m pip install cython # This dependency cannot be installed via brew.
  brew install bh107/homebrew-bohrium/bohrium


Manual install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO.

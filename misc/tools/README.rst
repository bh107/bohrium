======================
Notes on Visualization
======================

Creating a Trace
================

One can create a bytecode trace by::

    # Install a version of naive with debugging enabled
    cd <BH_SRC>/ve/naive

    # Make sure that INSTALLDIR points to where you install your bohrium libs...
    DEBUG="-DDEBUG" INSTALLDIR=~/bohrium.lib/ make clean all install

    # Then run whatever you want to get a graph of, such as
    cd BOHRIUM_SRC/misc/visualization/
    python ../../benchmark/Python/jacobi.iterative.py --size=10*10*10 --bohrium=True > traces/example.trace

Visualizing the trace: parse.py
===============================

Execute 'parse.py' to generate an image of the bytecode::

    ./parse.py traces/example.trace --output output/ --exclude FREE DISCARD

This will generate the file: "output/example.svg", try opening it with your browser.
parse.py has other options, inspect them with "-h".

Making the trace easier to read: symbolize.py
=============================================

The script simply renames memory-address to symbolic names for easier trace-debugging.
Can be invoked on a trace like so::

    ./symbolize.py < traces/example.trace

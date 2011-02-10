#!/bin/bash
#Bash script for building the NumPy bridge.

#Change current directory to where numpy and this script is located.
cd `dirname "$0"`

#Set Environment variables.
export CFLAGS="-I../../lib -I../../ve/svi -I../../util"
export LDFLAGS="-L../../lib -lcphvb -L../../ve/svi -lsvi -L../../util -lcphvbutil"

#Set Python Interpreter.
if [ $# = 1 ]
then
    PYTHON="$1"
else
    PYTHON="/usr/bin/python"
fi

#Call NumPy build script
$PYTHON setup.py build

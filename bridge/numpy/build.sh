#!/bin/bash
#Bash script for building the NumPy bridge.

#Set Environment variables.
CFLAGS="-I../../lib -I../../vm/svi -I../../util"
LDFLAGS="-L../../lib -lcphvb -L../../vm/svi -lsvi -L../../util -lcphvbutil"

#Set Python Interpreter.
if [ $# = 1 ]
then
    PYTHON="$1"
else
    PYTHON="/usr/bin/python"
fi

#Change current directory to where numpy and this script is located.
cd `dirname "$0"`

#Call NumPy build script
$PYTHON setup.py build

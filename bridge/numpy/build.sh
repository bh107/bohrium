#!/bin/bash
#Bash script for building the NumPy bridge.

#Change current directory to where numpy and this script is located.
cd `dirname "$0"`

#Set Environment variables.
export CFLAGS="-I../../include"
export LDFLAGS="-L../../core -lcphvb -L../../vem -lcphvem -L../../ve/simple -lcphvb_ve_simple"

#Set Python Interpreter.
if [ $# = 1 ]
then
    PYTHON="$1"
else
    PYTHON="/usr/bin/python"
fi

#Call NumPy build script
$PYTHON setup.py build

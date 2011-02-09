#!/bin/bash
#Bash script for building the NumPy bridge.

#set Environment variables.
CFLAGS="-I../../lib -I../../vm/svi -I../../util"
LDFLAGS="-L../../lib -lcphvb -L../../vm/svi -lsvi -L../../util -lcphvbutil"

#Change current directory to where numpy and this script is located.
cd `dirname "$0"`

#Call NumPy build script
python setup.py build

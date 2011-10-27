#!/bin/bash
#Bash script for building the NumPy bridge.

#Change current directory to where numpy and this script is located.
cd `dirname "$0"`

#Set the default environment variables.
ROOT="../.."
export CFLAGS="-I$ROOT/include -I$ROOT/iniparser/src"
export LDFLAGS="-L$ROOT/core -lcphvb $LDFLAGS"
export LDFLAGS="-L$ROOT/vem/node -lcphvb_vem_node $LDFLAGS"
export LDFLAGS="-L$ROOT/ve/simple -lcphvb_ve_simple $LDFLAGS"
PYTHON="/usr/bin/python"

while getopts "M:E:C:D:" opt; do
  case $opt in
    M)
      HIGH=`echo $OPTARG  | tr '[a-z]' '[A-Z]'`
      LOW=`echo $OPTARG  | tr '[A-Z]' '[a-z]'`
      export CFLAGS="-D$HIGH $CFLAGS"
      export LDFLAGS="-L$ROOT/vem/$LOW -lcphvb_vem_$LOW $LDFLAGS"
      ;;
    E)
      HIGH=`echo $OPTARG  | tr '[a-z]' '[A-Z]'`
      LOW=`echo $OPTARG  | tr '[A-Z]' '[a-z]'`
      export CFLAGS="-D$HIGH $CFLAGS"
      export LDFLAGS="-L$ROOT/ve/$LOW -lcphvb_ve_$LOW $LDFLAGS"
      if [ $HIGH = "CUDA" ] ; then
        export LDFLAGS="-L/usr/lib/nvidia -lcuda $LDFLAGS"
      fi
      ;;
    D)
      export CFLAGS="-D$OPTARG $CFLAGS"
      ;;
    C)
      PYTHON="$OPTARG"
      ;;
    \?)
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Decrements the argument pointer so it points to next argument.
# $1 now references the first non-option item supplied on the command-line
# if one exists.
shift $(($OPTIND - 1))

DEFAULT=1
for arg in $@
  do
    if [ "$arg" = "clean" ]
      then
        echo "rm -R build/"
        rm -R build/
        DEFAULT=0
    fi
    if [ "$arg" = "install" ]
      then
        $PYTHON setup.py build
        DEFAULT=0
    fi
done

if [ "$DEFAULT" -eq "1" ]
  then
    $PYTHON setup.py build
fi


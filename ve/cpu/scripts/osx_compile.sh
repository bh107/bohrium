#!/bin/bash

if [ ! -d "${TMPDIR}" ]; then
    TMPDIR=/tmp/
fi

TMPFILE=`mktemp "${TMPDIR}/bohrium.XXXXXX"`
if [ -f "${TMPFILE}" ]; then
    rm "${TMPFILE}"
fi

TMPFILE="${TMPFILE}".c

cat > "${TMPFILE}"
clang -arch x86_64 -arch i386 -lm -O3 -fPIC -std=c99 -x c "${TMPFILE}" -shared $@
rm "${TMPFILE}"

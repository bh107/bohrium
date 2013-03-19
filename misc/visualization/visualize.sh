#!/usr/bin/env bash
TMP_DOT=`mktemp -t XXXXXXXXXX.dot` || exit 1
TMP_SVG=`mktemp -t XXXXXXXXXX.svg` || exit 1
./parse.py > $TMP_DOT && dot -T svg $TMP_DOT > $TMP_SVG && xdg-open $TMP_SVG

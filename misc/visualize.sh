#!/usr/bin/env bash
#./parse.py > /tmp/test.dot && dot -T png /tmp/test.dot > test.png && eog test.png
./parse.py > /tmp/test.dot && dot -T svg /tmp/test.dot > test.svg && chromium-browser test.svg
#./parse.py > /tmp/test.dot && dot -T html /tmp/test.dot > test.html && chromium-browser test.html

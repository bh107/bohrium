#!/usr/bin/env bash
./parse.py > /tmp/test.dot && dot -T svg /tmp/test.dot > test.svg && chromium-browser test.svg

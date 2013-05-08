#!/usr/bin/env bash
killall gvimdiff
/usr/bin/time python ../../../benchmark/Python/black_scholes.py --size=1000*10 --bohrium=True | ../../../misc/tools/symbolize.py > bs_python.txt
/usr/bin/time ../../../benchmark/cpp/bin/black_scholes --size=1000*10 | ../../../misc/tools/symbolize.py > bs_cpp.txt
meld bs_python.txt bs_cpp.txt &

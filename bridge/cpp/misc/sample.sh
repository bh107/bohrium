#!/usr/bin/env bash
/usr/bin/time python ../../../benchmark/Python/black_scholes.py --size=1000*10 --bohrium=True | .././symbolize.py > bs_python.txt
/usr/bin/time ../../../bridge/cpp/bin/black_scholes | .././symbolize.py > bs_cpp.txt


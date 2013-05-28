#!/usr/bin/env bash
python ../../../benchmark/Python/black_scholes.py --size=1000000*10 --bohrium=True
python ../../../benchmark/Python/black_scholes.py --size=1000000*10 --bohrium=True
python ../../../benchmark/Python/black_scholes.py --size=1000000*10 --bohrium=True
../../../benchmark/cpp/bin/black_scholes  --size=1000000*10
../../../benchmark/cpp/bin/black_scholes  --size=1000000*10
../../../benchmark/cpp/bin/black_scholes  --size=1000000*10
../../../benchmark/blitz/bin/black_scholes   --size=1000000*10
../../../benchmark/blitz/bin/black_scholes   --size=1000000*10
../../../benchmark/blitz/bin/black_scholes   --size=1000000*10

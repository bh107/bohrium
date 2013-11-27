#!/usr/bin/env bash
rm /tmp/*.svg
rm *.dot
#BH_PRINT_NODE_INPUT_GRAPH="" ../../benchmark/cpp/bin/black_scholes --size=1000*1
find . -iname "*.dot" | while read FILE; do
dot -T svg $FILE -o /tmp/$FILE.svg
dot -T xdot $FILE -o /tmp/$FILE.xdot.gv
done;
xdg-open /tmp/input-graph-1.dot.svg
sleep 3
../../benchmark/cpp/bin/black_scholes --size=1000000*10
../../benchmark/cpp/bin/black_scholes --size=1000000*10
../../benchmark/cpp/bin/black_scholes --size=1000000*10

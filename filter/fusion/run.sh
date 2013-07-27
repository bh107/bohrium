#!/usr/bin/env bash
ulimit -c unlimited
rm /tmp/*.svg
rm *.dot
#BH_PRINT_NODE_INPUT_GRAPH="" ../../benchmark/cpp/bin/black_scholes --size=1000*1
BH_PRINT_NODE_INPUT_GRAPH="" ../../benchmark/cpp/bin/jacobi_stencil --size=1000*1000*1
#BH_PRINT_NODE_INPUT_GRAPH="" ../../benchmark/cpp/bin/monte_carlo_pi --size=1000*1
find . -iname "*.dot" | while read FILE; do
dot -T svg $FILE -o /tmp/$FILE.svg
dot -T xdot $FILE -o /tmp/$FILE.xdot.gv
done;
xdg-open /tmp/input-graph-1.dot.svg
sleep 3

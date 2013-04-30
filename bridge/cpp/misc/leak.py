#!/usr/bin/env python
import sys

#
# Leak from temporary re-assignment - FIXED
# 8byte leak from functions (sqrt, sin etc) - FIXED
# 8byte leak from REDUCE - FIXED
# 40bytes leak from RANDOM - FIXED
# 64bytes leak from REDUCE - FIXED
# Extension data should be de-allocated!
#
# It does however seem to leak something... or just not de-allocate it in a neat
# and efficient way.. perhaps the issue lies with dynamical view allocation...
#

def main(iterations):
    blocks_oh = 0
    blocks_pr = 3
    bytes_oh = 0
    bytes_pr = 40*2+64
    print "%d bytes in %d blocks" % (iterations*bytes_pr+bytes_oh,
                                     iterations*blocks_pr+blocks_oh)

if __name__ == "__main__":
    main(int(sys.argv[1]))

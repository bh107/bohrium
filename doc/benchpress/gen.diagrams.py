#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

from pylab import *
import json
import sys

def stats( times ):
    """ Returns: (avg, lowest, highest, deviation)"""
    
    return (sum(times)/float(len(times)), max(times), min(times), 0.0)

def main( argv ):

    output = 'gfx'

    raw     = json.load(open(argv[1]))
    meta    = raw['meta']
    runs    = raw['runs']

    bases = {}
    bench = {}
    for mark, engine_lbl, engine, engine_args, cmd, times in runs:

        t_avg, t_max, t_min, t_dev = stats(times)
        if engine:                                  # Results with cphvb enabled.

            if mark in bench:
                bench[mark][engine_lbl] = t_avg
            else:
                bench[mark] = { engine_lbl: t_avg }

        else:                                       # Without cphvb = baseline.
            bases[mark] = t_avg

    for mark in bench:
                                                    # Runtime in relation to baseline
        rt = [(engine_lbl, 1/(bases[mark]/bench[mark][engine_lbl])) for engine_lbl in (bench[mark]) ] 
        rt.sort()
        rt = rt[::-1]
        rt = [('numpy', bases[mark]/bases[mark])] + rt

                                                    # Speed-up in relation to baseline
        su = [(engine_lbl, (bases[mark]/bench[mark][engine_lbl])) for engine_lbl in (bench[mark]) ] 
        su.sort()
        su = su[::-1]
        su = [('numpy', bases[mark]/bases[mark])] + su

        graphs = [
            ('Speedup', su),
            ('Runtime', rt),
        ]

        for graph, data in graphs:

            lbl = [engine_lbl for engine_lbl, time in data]
            val = [time for engine_lbl, time in data]
            pos = arange(len(val))

            figure(1)
            bar(pos, val, align='center')

            ylabel('%s in relation to NumPy' % graph)
            xticks(pos, lbl)
            xlabel('Vector Engine')
            title(mark)
            grid(True)
            
            savefig("%s/%s_%s.pdf" % ( output, mark.lower(), graph.lower() ))
            savefig("%s/%s_%s.eps" % ( output, mark.lower(), graph.lower() ))
            savefig("%s/%s_%s.png" % ( output, mark.lower(), graph.lower() ))
            show()

if __name__ == "__main__":
    main( sys.argv )

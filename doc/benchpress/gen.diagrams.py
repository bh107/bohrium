#!/usr/bin/env python
from pylab import *
import json
import sys

def main( argv ):

    output = 'gfx'

    data = json.load(open(argv[1]))

    bench = {}
    bases = {}
    names = {
        'jacobi_fixed': 'Jacobi Solver', 
        'kNN':          'kNN', 
        'stencil':      'Synthetic Stencil',
        'swater':       'Shallow Water'
    }
                                                    # Sort out the best numpy time from the runs
    for mark, cphvb, engine, run, time in data:

        if cphvb:       # skip times from cphVB
            continue

        mark = mark.split('.')[0]
        if mark in bases:
            if time < bases[mark]:
                bases[mark] = time
        else:
            bases[mark] = time

    for mark, cphvb, engine, run, time in data:     # Sort out the best cphvb time from the runs

        if not cphvb:   # skip times from numpy
            continue

        mark = mark.split('.')[0]
        if mark in bench:
            engine = engine if cphvb else 'numpy'
            if engine in bench[mark]:
                if time < bench[mark][engine]:
                    bench[mark][engine] = time
            else:
                bench[mark][engine] = time
        else:
            bench[mark] = {engine: time }

    for mark in bench:
                                                        # Runtime in relation to NumPy
        rt = [(engine, 1/(bases[mark]/bench[mark][engine])) for engine in (bench[mark]) ] 
        rt.sort()
        rt = rt[::-1]
        rt = [('numpy', bases[mark]/bases[mark])] + rt
                                                        # Speed-up in relation to NumPy
        su = [(engine, (bases[mark]/bench[mark][engine])) for engine in (bench[mark]) ] 
        su.sort()
        su = su[::-1]
        su = [('numpy', bases[mark]/bases[mark])] + su

        graphs = [
            ('Speedup', su),
            ('Runtime', rt),
        ]

        for graph, data in graphs:

            lbl = [engine for engine, time in data]
            val = [time for engine, time in data]
            pos = arange(len(val))

            figure(1)
            bar(pos, val, align='center')

            ylabel('%s in relation to NumPy' % graph)
            xticks(pos, lbl)
            xlabel('Vector Engine')
            title(names[mark])
            grid(True)
            
            savefig("%s/%s_%s.pdf" % ( output, mark.lower(), graph.lower() ))
            savefig("%s/%s_%s.eps" % ( output, mark.lower(), graph.lower() ))
            savefig("%s/%s_%s.png" % ( output, mark.lower(), graph.lower() ))
            show()

if __name__ == "__main__":
    main( sys.argv )

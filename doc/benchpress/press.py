#!/usr/bin/env python
from ConfigParser import SafeConfigParser
from subprocess import Popen, PIPE
import tempfile
import json
import os

def main(config):

    script_path= '../../benchmark/Python/'
    out_path='./results'

    parser = SafeConfigParser()
    parser.read(config)

    engines = [('simple',False,1) ]
    #engines = [('simple',False,3), ('simple',True, 3), ('score',True, 3), ('mcore',True, 3)]
    bench   = [
        ('jacobi_fixed.py', '--size=7168*7168*4'),
        ('MonteCarlo.py',   '--size=100000000*1'),
        ('swater.py',       '--size=3600*1'),
        ('stencil.py',      '--size=10240*1024*10'),
        ('kNN.py',          '--size=10000*120')
    ]
    
    #run = [0]
    run = [0,2,3,4]

    # Not running monte-carlo since it is cphvb_reduce cannot currently handle
    # the mixed type operation which it generates.

    times = []

    for r in run:
        for engine, cphvb, runs in engines:
            parser.set("node", "children", engine)
            parser.write(open(config, 'wb'))

            args = ['python', script_path+bench[r][0], bench[r][1], '--cphvb=%s' % cphvb ]
            print '-{[',engine,',', cphvb,',',' '.join(args[1:]), '.'
            for i in xrange(1,runs+1):
                p = Popen(
                    args,
                    stdin=PIPE,
                    stdout=PIPE
                )
                out, err = p.communicate()
                print "RUN",i, out, err, out.split(' ')[-1]
                text = ' '.join(bench[r])
                times.append( (text, cphvb, engine, i, float(out.split(' ')[-1] .rstrip()) ))

            print "]}-"

    f = tempfile.NamedTemporaryFile(delete=False, dir=out_path, prefix='benchmark-')
    json.dump(times, f, indent=4)

if __name__ == "__main__":
    main(os.getenv('HOME')+os.sep+'.cphvb'+os.sep+'config.ini')


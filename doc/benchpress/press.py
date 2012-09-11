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

    engines = [
        ('simple', 'numpy',         False,3, None),
        ('simple', 'simple',        True, 3, None),
        ('score',  'score_1',       True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"1"}),
        ('score',  'score_32',      True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"32"}),
        ('score',  'score_64',      True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"64"}),
        ('score',  'score_128',     True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"128"}),
        ('score',  'score_512',     True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"512"}),
        ('score',  'score_1024',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"1024"}),
        ('mcore',  'mcore',         True, 3, None)
    ]
    bench   = [
        ('cache.py',        '--size=10485760*10*1'),
        ('jacobi_fixed.py', '--size=7168*7168*4'),
        ('MonteCarlo.py',   '--size=100000000*1'),
        ('swater.py',       '--size=3600*1'),
        ('stencil.py',      '--size=10240*1024*10'),
        ('kNN.py',          '--size=10000*120')
    ]
    
    run     = [0]
    using   = [0,1,2,3]
    #run = [0,2,3,4]

    # Not running monte-carlo since it is cphvb_reduce cannot currently handle
    # the mixed type operation which it generates.

    times = []

    for r in run:
        for engine, engine_str, cphvb, runs, env in engines:
            parser.set("node", "children", engine)
            parser.write(open(config, 'wb'))

            envs = None
            if env:
                envs = os.environ.copy()
                envs.update(env)

            args = ['python', script_path+bench[r][0], bench[r][1], '--cphvb=%s' % cphvb ]
            print '-{[',engine,',', cphvb,',',' '.join(args[1:]), '.'
            for i in xrange(1,runs+1):
                p = Popen(
                    args,
                    stdin=PIPE,
                    stdout=PIPE,
                    env=envs
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


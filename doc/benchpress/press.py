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
        ('score',  'score_16',      True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"32"}),
        ('score',  'score_64',      True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"64"}),
        ('score',  'score_512',     True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"512"}),
        ('score',  'score_1024',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"1024"}),
        ('score',  'score_2048',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"2048"}),
        ('score',  'score_4096',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"2048"}),
        ('score',  'score_8192',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"8192"}),
        ('score',  'score_16384',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"16384"}),
        ('score',  'score_32768',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"32768"}),
        ('score',  'score_65536',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"65536"}),
        ('score',  'score_131072',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"131072"}),
        ('score',  'score_262144',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"262144"}),
        ('score',  'score_524288',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"524288"}),
        ('score',  'score_1048576',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"1048576"}),
        ('score',  'score_2097152',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"2097152"}),
        ('score',  'score_4194304',    True, 3, {"CPHVB_VE_SCORE_BLOCKSIZE":"4194304"}),
    ]
    bench   = [
        ('simplest.py',     '--size=100000000*1'),
        ('cache.py',        '--size=10485760*10*1'),
        ('jacobi_fixed.py', '--size=7168*7168*4'),
        ('MonteCarlo.py',   '--size=100000000*1'),
        ('swater.py',       '--size=3600*1'),
        ('stencil.py',      '--size=10240*1024*10'),
        ('twonine.py',      '--size=10240*1024*10'),
        ('kNN.py',          '--size=10000*120')
    ]
    
    run     = [0,1,2,3,4,5,6,7]
    using   = [0,1,2,3,4]

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
                try:
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
                except:
                    print "Error running benchmark:", sys.exc_info()[0]
            print "]}-"

    f = tempfile.NamedTemporaryFile(delete=False, dir=out_path, prefix='benchmark-')
    json.dump(times, f, indent=4)

if __name__ == "__main__":
    main(os.getenv('HOME')+os.sep+'.cphvb'+os.sep+'config.ini')


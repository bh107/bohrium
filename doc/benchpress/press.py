#!/usr/bin/env python
from ConfigParser import SafeConfigParser
from subprocess import Popen, PIPE
from datetime import datetime
import tempfile
import json
import os

def meta():

    return {
        'date': str(datetime.now()),
        'cpu':  open('/proc/cpuinfo','r').read(),
        'os':   open('/proc/version','r').read()
    }

def main(config):

    script_path = '../../benchmark/Python/'.replace('/', os.sep)
    out_path    ='./results'

    # Engines with various parameter setups
    # (alias, runs, engine, env-vars)
    runs    = 5
    engines = [
        ('numpy',       runs,  None,        None),
        ('simple',      runs,  'simple',    None),
        ('score',       runs,  'score', None),
        ('mcore',       runs,  'mcore', None),

        ('score_1',     runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"1"}),
        ('score_2',     runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"2"}),
        ('score_4',     runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"4"}),
        ('score_16',     runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"16"}),
        ('score_32',    runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"32"}),
        ('score_64',    runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"64"}),
        ('score_512',   runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"512"}),
        ('score_1024',  runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"1024"}),
        ('score_2048',  runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"2048"}),
        ('score_4096',  runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"4096"}),
        ('score_8192',  runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"8192"}),
        ('score_16384', runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"16384"}),
        ('score_32768', runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"32768"}),
        ('score_65536', runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"65536"}),
        ('score_131072',    runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"131072"}),
        ('score_262144',    runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"262144"}),
        ('score_524288',    runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"524288"}),
        ('score_1048576',   runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"1048576"}),
        ('score_2097152',   runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"2097152"}),
        ('score_4194304',   runs,  'score',    {"CPHVB_VE_SCORE_BLOCKSIZE":"4194304"}),
    ]

    # Scripts and their arguments
    # (alias, script, parameters)
    scripts   = [
        ('Jacobi Fixed',    'jacobi_fixed.py', '--size=7168*7168*4'),
        ('Monte Carlo',     'MonteCarlo.py',   '--size=100000000*1'),
        ('Shallow Water',   'swater.py',       '--size=2200*1'),
        ('kNN',             'kNN.py',          '--size=10000*120'),
        ('Stencil Synth',   'stencil.py',      '--size=10240*1024*10'),

        ('Cache Synth',     'cache.py',        '--size=10485760*10*1'),
        ('Stencil Synth2',  'twonine.py',      '--size=10240*1024*10'),
        ('Stencil Synth3',  'simplest.py',     '--size=100000000*1')
    ]
    
    benchmark = {                   # Define a benchmark which runs
        'scripts': [7],             # these scripts
        'engines': [12]             # using these engines
    }   
                                    # DEFAULT BENCHMARK
    benchmark = {                   # Define a benchmark which runs
        'scripts': [0,1,2,3,4],     # these scripts
        'engines': [0,1,2,3]        # using these engines
    } 

    parser = SafeConfigParser()     # Parser to modify the cphvb configuration file.
    parser.read(config)             # Read current configuration

    results = {
        'meta': meta(),
        'runs': []
    }
    with tempfile.NamedTemporaryFile(delete=False, dir=out_path, prefix='benchmark-') as fd:
        print "Benchmarks are written to: %s." % fd.name
        for mark, script, arg in (scripts[snr] for snr in benchmark['scripts']):
            for alias, runs, engine, env in (engines[enr] for enr in benchmark['engines']):

                cphvb = False
                if engine:                                  # Enable cphvb with the given engine.
                    cphvb = True
                    parser.set("node", "children", engine)  
                    parser.write(open(config, 'wb'))

                envs = None                                 # Populate environment variables
                if env:
                    envs = os.environ.copy()
                    envs.update(env)
                                                            # Setup process + arguments
                args        = ['python', script_path + script, arg, '--cphvb=%s' % cphvb ]
                args_str    = ' '.join(args)
                print "{ %s - %s ( %s ),\n  %s" % ( mark, alias, engine, args_str )

                times = []
                for i in xrange(1, runs+1):

                    p = Popen(                              # Run the command
                        args,
                        stdin=PIPE,
                        stdout=PIPE,
                        env=envs
                    )
                    out, err = p.communicate()              # Grab the output
                    elapsed = 0.0
                    if err:
                        print "ERR: Something went wrong %s" % err
                    else:
                        elapsed = float(out.split(' ')[-1] .rstrip())

                    print "  %d/%d, " % (i, runs), elapsed
                    
                    times.append( elapsed )

                print "}"
                                                            # Accumulate results
                results['runs'].append(( mark, alias, engine, env, args_str, times ))

                fd.truncate(0)                              # Store the results in a file...
                fd.seek(0)
                fd.write(json.dumps(results, indent=4))
                fd.flush()
                os.fsync(fd)

if __name__ == "__main__":
    main(os.getenv('HOME')+os.sep+'.cphvb'+os.sep+'config.ini')


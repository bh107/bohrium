#!/usr/bin/python
#Benchmarks for DistNumPy.
#This is collection of help functions for the DistNumPy benchmarks.

import numpy as np
import cphvbnumpy
import getopt
import sys
import datetime
import time
from os import environ as env
import os
import multiprocessing
import subprocess
import pickle

class Benchmark:
    """This class should handle the presentation of benchmark results.
       A list of non-optional arguments is exposed through self.argv.
    """
    def __init__(self):
        self.batch_mode = False
        self.info = {'cphvb':False, 'date':datetime.datetime.now(),'file':os.path.basename(sys.argv[0])}
        options, self.argv = getopt.gnu_getopt(sys.argv[1:], \
                'p:n:c:s:',\
                ['cphvb=','nnodes=','ncores=','size=','batch'])

        for opt, arg in options:
            if opt in ('-p', '--cphvb'):
                self.info['cphvb'] = bool(eval(arg))
            if opt in ('-n', '--nnodes'):
                self.info['nnodes'] = int(arg)
            if opt in ('-c', '--ncores'):
                self.info['ncores'] = int(arg)
            if opt in ('--batch'):
                self.batch_mode = True
            if opt in ('--size'):
                #Jobsize use the syntax: dim_size*dim_size fx. 10*20
                self.info['size'] = [int(i) for i in arg.split("*") if len(i)]

        self.info['nthd'] = multiprocessing.cpu_count()
        self.info['nblocks'] = 16
        try:
            self.info['nthd'] = int(env['OMP_NUM_THREADS'])
        except KeyError:
            pass
        try:
            self.info['nblocks'] = int(env['CPHVB_SCORE_NBLOCKS'])
        except KeyError:
            pass
        #Expose variables to the user.
        self.size  = self.info['size']
        self.cphvb = self.info['cphvb']

    def start(self):
        self.info['totaltime'] = time.time()

    def stop(self):
        cphvbnumpy.flush()
        self.info['totaltime'] = time.time() - self.info['totaltime']

    def pprint(self):
        if self.batch_mode:
            print "%s"%pickle.dumps(self.info)
        else:
            print "%s - cphvb: %s, nthd: %d, nblocks: %d size: %s, total time: %f"%(self.info['file'],self.info['cphvb'],self.info['nthd'],self.info['nblocks'],self.info['size'],self.info['totaltime'])


if __name__ == "__main__":
    nblocks = 16
    options, remainders = getopt.gnu_getopt(sys.argv[1:], '', ['file=','thd-min=', 'thd-max=', 'jobsize=','repeat=','seq', 'nblocks='])
    for opt, arg in options:
        if opt in ('--file'):
            filename = arg
        if opt in ('--thd-min'):
            minthd = int(arg)
        if opt in ('--thd-max'):
            maxthd = int(arg)
        if opt in ('--jobsize'):
            jobsize = arg
        if opt in ('--repeat'):
            repeat = int(arg)
        if opt in ('--nblocks'):
            nblocks = int(arg)
        if opt in ('--seq'):
            cphvb = False
    cphvb = False
    nthd = minthd
    while nthd <= maxthd:
        try:
            env = os.environ
            env['OMP_NUM_THREADS'] = "%d"%nthd
            env['CPHVB_SCORE_NBLOCKS'] = "%d"%nblocks
            p = subprocess.Popen([sys.executable,filename,"--batch","--cphvb=%s"%cphvb, "--size",jobsize],env=env,stdout=subprocess.PIPE)
            (stdoutdata, stderrdata) = p.communicate()
            info = pickle.loads(stdoutdata)
            if nthd == minthd and not cphvb:#First iteration
                print "#%s"%info
                print "#NumPy;%10.4f\n"%(info['totaltime'])
            else:
                print "%6.d;%10.4f\n"%(nthd,info['totaltime'])
            err = p.wait()
        except KeyboardInterrupt:
            p.terminate()

        if nthd == minthd and not cphvb:#First iteration
            cphvb = True
        else:
            nthd *= 2

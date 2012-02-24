#Benchmarks for DistNumPy.
#This is collection of help functions for the DistNumPy benchmarks.

import numpy as np
import getopt
import sys
import datetime
import time
from os import environ as env
import os
import multiprocessing

class Benchmark:
    """This class should handle the presentation of benchmark results.
       A list of non-optional arguments is exposed through self.argv.
    """
    def __init__(self):
        self.info = {'cphvb':False, 'date':datetime.datetime.now()}
        options, self.argv = getopt.gnu_getopt(sys.argv[1:], \
                'p:n:c:s:',\
                ['cphvb=','nnodes=','ncores=','size='])

        for opt, arg in options:
            if opt in ('-p', '--cphvb'):
                self.info['cphvb'] = bool(eval(arg))
            if opt in ('-n', '--nnodes'):
                self.info['nnodes'] = int(arg)
            if opt in ('-c', '--ncores'):
                self.info['ncores'] = int(arg)
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
        self.totaltime = time.time()

    def stop(self):
        self.totaltime = time.time() - self.totaltime

    def pprint(self):
        #print self.info
        print "%s - cphvb: %s, nthd: %d, nblocks: %d size: %s, total time: %f"%(os.path.basename(sys.argv[0]),self.info['cphvb'],self.info['nthd'],self.info['nblocks'],self.info['size'],self.totaltime)



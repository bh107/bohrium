#Test and demonstration of the NumPy Bridge.
from __future__ import print_function

from operator import mul
from numbers import Number
import subprocess
import warnings
import random
import pickle
import time
import uuid
import copy
import sys
import os
from os.path import join
import re
import argparse

import numpy as np
import bohrium as bh
from functools import reduce

class TYPES:
    NORMAL_INT   = ['np.int32','np.int64','np.uint32','np.uint64']
    ALL_INT      = NORMAL_INT + ['np.int8','np.int16','np.uint8','np.uint16']
    NORMAL_FLOAT = ['np.float32','np.float64']
    ALL_FLOAT    = NORMAL_FLOAT #+ ['np.float16'] float16 is only supported by the GPU
    NORMAL       = NORMAL_INT + NORMAL_FLOAT
    ALL          = ALL_INT + ALL_FLOAT

class _C:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def gen_shapes(max_ndim, max_dim, iters=0, min_ndim=1):
    for ndim in xrange(min_ndim,max_ndim+1):
        shape = [1]*ndim
        if iters:
            yield shape #Min shape
            yield [max_dim]*(ndim) #Max shape
            for _ in xrange(iters):
                for d in xrange(len(shape)):
                    shape[d] = np.random.randint(1,max_dim)
                yield shape
        else:
            finished = False
            while not finished:
                yield shape
                #Find next shape
                d = ndim-1
                while True:
                    shape[d] += 1
                    if shape[d] > max_dim:
                        shape[d] = 1
                        d -= 1
                        if d < 0:
                            finished = True
                            break
                    else:
                        break

def gen_views(max_ndim, max_dim, iters=0, min_ndim=1):
    for shape in gen_shapes(max_ndim, max_dim, iters, min_ndim):
        #Base array
        cmd = "a[0] = self.array(%s,np.float32);"%(shape)
        yield cmd
        #Views with offset per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += "1:,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

        #Views with negative offset per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += ":-1,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

        #Views with steps per dimension
        for d in xrange(len(shape)):
            if shape[d] > 1:
                s = "a[0] = a[0]["
                for _ in xrange(d):
                    s += ":,"
                s += "::2,"
                for _ in xrange(len(shape)-(d+1)):
                    s += ":,"
                s = s[:-1] + "];"
                yield cmd + s

class numpytest:
    def __init__(self):
        self.config = {'maxerror':0.0}
        self.runtime = {}
        self.random = random.Random()
        self.random.seed(42)

    def init(self):
        pass
    def array(self,dims,dtype,high=False):
        try:
            total = reduce(mul,dims)
        except TypeError:
            total = dims
            dims = (dims,)
        dtype = np.dtype(dtype).type
        if dtype is np.bool:
            res = np.random.random_integers(0,1,dims)
        elif dtype in [np.int8, np.uint8]:
            res = np.random.random_integers(1,3,dims)
        elif dtype is np.int16:
            res = np.random.random_integers(1,5,dims)
        elif dtype is np.uint16:
            res = np.random.random_integers(1,6,dims)
        elif dtype in [np.float32, np.float64]:
            res = np.random.random(size=dims)
            if high:
                res = (res+1)*10
        elif dtype in [np.complex64, np.complex128]:
            res = np.random.random(size=dims)+np.random.random(size=dims)*1j
        else:
            res = np.random.random_integers(1,8,size=dims)
        if len(res.shape) == 0:#Make sure scalars is arrays.
            res = np.asarray(res)
            res.shape = dims
        return np.asarray(res, dtype=dtype)

class BenchHelper:
    """Mixin for numpytest to aid the execution of Benchmarks."""

    def init(self):
        """
        This function is used as a means to control til --dtype argument
        passed to the benchmark script and provide a uuid for benchmark output.
        """
        self.uuid = str(uuid.uuid4())
        for dtype in self.dtypes:
            yield ({0:bh.empty(self.size, bohrium=False, dtype=dtype)},
                   "%s: " % str(dtype)
            )

    def get_meta(self, arrays):
        """Determine target and dtype based on meta-data from pseudo_init."""

        target = "None"
        if 'bohrium.ndarray' in str(type(arrays[0])):
            target = "bhc"

        dtype = str(arrays[0].dtype)

        return (target, dtype)

    def run(self, pseudo_input):
        """
        Run the Benchmark script and return the result.

        Benchmarks are assumed to be installed along with the Bohrium module.
        """

        (target, dtype) = self.get_meta(pseudo_input)

        # Setup output filename
        outputfn = "/tmp/%s_%s_%s_output_%s.npz" % (
            self.script, dtype, target, self.uuid
        )

        bench_dir = join(os.path.dirname(os.path.realpath(__file__)),"..","..","benchmark","python")
        # Setup command
        cmd = [
            sys.executable, #The current Python interpreter
            join(bench_dir,"%s.py"%self.script),
            '--size='       +self.sizetxt,
            '--dtype='      +str(dtype),
            '--target='    +target,
            '--outputfn='   +outputfn
        ]

        # Setup the inputfn if one is needed/provided
        if self.inputfn:
            npt_path = os.path.dirname(sys.argv[0])
            if not npt_path:
                npt_path = "./"

            inputfn = "%s/datasets/%s" % (
                npt_path,
                self.inputfn.format(dtype)
            )
            cmd.append('--inputfn')
            cmd.append(inputfn)

            if not os.path.exists(inputfn):
                raise Exception('File does not exist: %s' % inputfn)

        p = subprocess.Popen(           # Execute the benchmark
            cmd,
            stdout  = subprocess.PIPE,
            stderr  = subprocess.PIPE
        )
        out, err = p.communicate()
        if 'elapsed-time' not in out:
            raise Exception("Benchmark error [stdout:%s,stderr:%s]" % (out, err))
        if err and not re.match("\[[0-9]+ refs\]", err): #We accept the Python object count
            raise Exception("Benchmark error[%s]" % err)

        if not os.path.exists(outputfn):
            raise Exception('Benchmark did not produce the output: %s' % outputfn)

        npzs    = np.load(outputfn)     # Load the result from disk
        res     = {}
        for k in npzs:
            res[k] = npzs[k]
        del npzs                        # Delete npz

        if os.path.exists(outputfn):    # Delete the result from disk
            os.remove(outputfn)

        # Convert to whatever namespace it ought to be in
        res['res'] = bh.array(res['res'], bohrium=target!="None")

        return (res['res'], ' '.join(cmd))

if __name__ == "__main__":
    warnings.simplefilter('error')#Warnings will raise exceptions
    pydebug = True

    try:
        sys.gettotalrefcount()
    except AttributeError:
        pydebug = False

    parser = argparse.ArgumentParser(description='Runs the test suite, which consist of all the test_*.py files')
    parser.add_argument(
        '--file',
        type=str,
        action='append',
        default=[],
        help='Add test file (supports multiple use of this argument)'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        action='append',
        default=[],
        help='Exclude test file (supports multiple use of this argument)'
    )
    parser.add_argument(
        '--test',
        type=str,
        action='append',
        default=[],
        help='Only run a specific test method '\
             '(supports multiple use of this argument)'
    )
    parser.add_argument(
        '--exclude-test',
        type=str,
        action='append',
        default=[],
        help='Only run a specific test method '\
             '(supports multiple use of this argument)'
    )
    parser.add_argument(
        '--exclude-benchmarks',
        action='store_true',
        help='Excludes all benchmak tests'
    )
    args = parser.parse_args()
    if len(args.file) == 0:
        args.file = os.listdir(os.path.dirname(os.path.abspath(__file__)))

    print("*"*3, "Testing the equivalency of Bohrium-NumPy and NumPy", "*"*3)
    for f in args.file:

        if f.startswith("test_") and f.endswith("py") and f not in args.exclude:
            m = f[:-3]#Remove ".py"
            m = __import__(m)
            #All test classes starts with "test_"
            for cls in [o for o in dir(m) if o.startswith("test_") and \
                        (True if args.test and o in args.test or not args.test else False)]:
                if cls in args.exclude_test:
                    continue

                cls_obj  = getattr(m, cls)
                cls_inst = cls_obj()

                import inspect
                is_benchmark = BenchHelper.__name__ in [c.__name__ for c in inspect.getmro(cls_obj)]
                if args.exclude_benchmarks and is_benchmark:
                    continue

                #All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    name = "%s/%s/%s"%(f,cls[5:],mth[5:])
                    print("Testing %s"%(name))
                    for (np_arys,cmd) in getattr(cls_inst,"init")():
                        #Get Bohrium arrays
                        bh_arys = []
                        for a in np_arys.values():
                            bh_arys.append(bh.array(a))
                        #Execute using NumPy
                        (res1,cmd1) = getattr(cls_inst,mth)(np_arys)
                        res1 = res1.copy()

                        #Execute using Bohrium
                        (res2,cmd2) = getattr(cls_inst,mth)(bh_arys)
                        cmd += cmd1
                        try:
                            if not np.isscalar(res2):
                                res2 = res2.copy2numpy()
                        except RuntimeError as error_msg:
                            print(_C.OKBLUE + "[CMD]   %s"%cmd + _C.ENDC)
                            print(_C.FAIL + str(error_msg) + _C.ENDC)
                        else:
                            rtol = cls_inst.config['maxerror']
                            atol = rtol * 0.1
                            if not np.allclose(res1, res2, rtol=rtol, atol=atol):
                                print(_C.FAIL + "[Error] %s"%(name) + _C.ENDC)
                                print(_C.OKBLUE + "[CMD]   %s"%cmd + _C.ENDC)
                                print(_C.OKGREEN + str(res1) + _C.ENDC)
                                print(_C.FAIL + str(res2) + _C.ENDC)
                                sys.exit (1)

    print("*"*24, "Finish", "*"*24)

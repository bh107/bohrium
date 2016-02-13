# -*- coding: utf-8 -*-
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
    SIGNED_INT   = ['np.int8', 'np.int16', 'np.int32','np.int64']
    UNSIGNED_INT = list(set(ALL_INT) - set(SIGNED_INT))
    COMPLEX      = ['np.complex64', 'np.complex128']
    NORMAL_FLOAT = ['np.float32','np.float64']
    ALL_FLOAT    = NORMAL_FLOAT #+ ['np.float16'] float16 is only supported by the GPU
    ALL_SIGNED   = SIGNED_INT + ALL_FLOAT + COMPLEX
    NORMAL       = NORMAL_INT + NORMAL_FLOAT
    ALL          = ALL_INT + ALL_FLOAT + COMPLEX

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

def gen_views(max_ndim, max_dim, iters=0, min_ndim=1, dtype="np.float32"):
    for shape in gen_shapes(max_ndim, max_dim, iters, min_ndim):
        #Base array
        cmd = "a[0] = self.array(%s,%s);"%(shape, dtype)
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

    def asarray(self, data, dtype):
        return np.asarray(data, dtype=dtype)

    def ones(self, shape, dtype):
        return np.asarray(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def arange(self, begin, end, step, dtype):
        return np.arange(begin, end, step, dtype=dtype)

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

def shell_cmd(cmd, cwd=None, verbose=False, env=None):

    from subprocess import Popen, PIPE, STDOUT
    if verbose:
        cmd.append('--verbose')
    cmd = " ".join(cmd)
    if verbose:
        print (cmd)
    out = ""
    try:
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True, cwd=cwd, env=env)
        while p.poll() is None:
            t = p.stdout.readline()
            out += t
            if verbose:
                print (t,)
        t = p.stdout.read()
        out += t
        if verbose:
            print (t,)
        p.wait()
    except KeyboardInterrupt:
        p.kill()
        raise
    return out

class BenchHelper:
    """Mixin for numpytest to aid the execution of Benchmarks."""

    def init(self):
        """
        This function is used as a means to control til --dtype argument
        passed to the benchmark script and provide a uuid for benchmark output.
        """
        #Lets make sure that benchpress is installed
        try:
            p = subprocess.Popen(
                ['bp-info', '--benchmarks'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            out, err = p.communicate()
            rc = p.returncode
        except OSError:
            print("ERROR: benchpress is not installed -- skipping test.")
            raise StopIteration()

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

        benchmarks_dir, err = subprocess.Popen(           # Execute the benchmark
            ['bp-info', '--benchmarks'],
            stdout  = subprocess.PIPE,
            stderr  = subprocess.PIPE,
        ).communicate()

        sys_exec = [sys.executable] if target.lower() == "none" else [sys.executable, "-m", "bohrium"]
        benchmark_path = os.sep.join([benchmarks_dir.strip(), self.script, "python_numpy", self.script+ ".py"])

        # Setup command
        cmd = sys_exec + [
            benchmark_path,
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

        env = os.environ.copy()
        env['BH_PROXY_PORT'] = "4201"
        out = shell_cmd(cmd, verbose=self.args.verbose, env=env) # Execute the benchmark
        if 'elapsed-time' not in out:
            raise Exception("Cannot find elapsed time, output:\n%s\n\n" %out)

        if not os.path.exists(outputfn):
            raise Exception('Benchmark did not produce any output, expected: %s' % outputfn)

        npzs    = np.load(outputfn)     # Load the result from disk
        res     = {}
        for k in npzs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    parser.add_argument(
        '--exclude-complex-dtype',
        action='store_true',
        help="Exclude tests with arrays of complex dtype"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print benchmark output"
    )
    parser.add_argument(
        '--no-complex128',
        action='store_true',
        help="Disable complex128 tests"
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

                if cls in args.exclude_test:            # Exclude specific test
                    continue

                cls_obj  = getattr(m, cls)
                cls_inst = cls_obj()
                cls_inst.args = args

                import inspect                          # Exclude benchmarks
                is_benchmark = BenchHelper.__name__ in [c.__name__ for c in inspect.getmro(cls_obj)]
                if args.exclude_benchmarks and is_benchmark:
                    continue

                testOkay = True
                # All test methods starts with "test_"
                for mth in [o for o in dir(cls_obj) if o.startswith("test_")]:
                    name = "%s/%s/%s"%(f,cls[5:],mth[5:])
                    print("Testing " + _C.OKGREEN + str(name) + _C.ENDC, end=" ")
                    sys.stdout.flush()

                    for (np_arys, cmd) in getattr(cls_inst,"init")():
                        if args.exclude_complex_dtype:  # Exclude complex
                            complex_nptypes = [eval(dtype) for dtype in TYPES.COMPLEX]

                            index = 0
                            non_complex = {}
                            for ary in np_arys.values():
                                if ary.dtype not in complex_nptypes:
                                    non_complex[index] = ary
                                    index += 1
                            np_arays = non_complex

                        bh_arys = []                    # Get Bohrium arrays
                        for a in np_arys.values():
                            bh_arys.append(bh.array(a))
                                                        # Execute using NumPy
                        (res1,cmd1) = getattr(cls_inst,mth)(np_arys)
                        res1 = res1.copy()
                                                        # Execute using Bohrium
                        (res2,cmd2) = getattr(cls_inst,mth)(bh_arys)
                        cmd += cmd1
                        try:                            # Compare
                            if not np.isscalar(res2):
                                res2 = res2.copy2numpy()
                        except RuntimeError as error_msg:
                            testOkay = False
                            print()
                            print("  " + _C.OKBLUE + "[CMD]   %s"%cmd + _C.ENDC)
                            print("  " + _C.FAIL   + str(error_msg)  + _C.ENDC)
                        else:
                            rtol = cls_inst.config['maxerror']
                            atol = rtol * 0.1
                            if not np.allclose(res1, res2, rtol=rtol, atol=atol):
                                testOkay = False
                                if 'warn_on_err' in cls_inst.config:
                                    print()
                                    print(_C.WARNING + "  [Warning] %s"%(name)                    + _C.ENDC)
                                    print(_C.OKBLUE  + "  [CMD]     %s"%cmd                       + _C.ENDC)
                                    print(_C.OKGREEN + "  NumPy result:   %s"%(res1)              + _C.ENDC)
                                    print(_C.FAIL    + "  Bohrium result: %s"%(res2)              + _C.ENDC)
                                    print(_C.WARNING + "  " + str(cls_inst.config['warn_on_err']) + _C.ENDC)
                                    print(_C.OKBLUE  + "  Manual verification is needed."         + _C.ENDC)
                                else:
                                    print()
                                    print(_C.FAIL    + "  [Error] %s"%(name)         + _C.ENDC)
                                    print(_C.OKBLUE  + "  [CMD]   %s"%cmd            + _C.ENDC)
                                    print(_C.OKGREEN + "  NumPy result:   %s"%(res1) + _C.ENDC)
                                    print(_C.FAIL    + "  Bohrium result: %s"%(res2) + _C.ENDC)
                                    sys.exit(1)
                    if testOkay: print("✓")

    print("*"*24, "Finish", "*"*24)

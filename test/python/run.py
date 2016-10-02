# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import bohrium
import numpy
import sys
import os
import imp

# Terminal colors
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def get_test_object_names(obj):
    """Returns all attribute names that starts with "test_"""
    ret = []
    for o in dir(obj):
        if o.startswith("test_"):
            ret.append(o)
    return ret


def run(args):
    for filename in args.files:
        if not filename.endswith("py"):
            continue # Ignore non-python files
        module_name = os.path.basename(filename)[:-3]  # Remove ".py"
        m = imp.load_source(module_name, filename)
        for cls_name in get_test_object_names(m):
            cls_obj = getattr(m, cls_name)
            cls_inst = cls_obj()
            for mth_name in get_test_object_names(cls_obj):
                mth_obj = getattr(cls_inst, mth_name)
                name = "%s/%s/%s" % (filename, cls_name[5:], mth_name[5:])
                print("Testing %s%s%s" % (OKGREEN, name, ENDC), end="")
                start_time = time.time()
                for ret in getattr(cls_inst, "init")():
                    # Let's retrieve the NumPy and Bohrium commands
                    cmd = mth_obj(ret)
                    if len(cmd) == 2:
                        (cmd_np, cmd_bh) = cmd
                    else:  # if not returning a pair, the NumPy and Bohrium command are identical
                        cmd_np = cmd
                        cmd_bh = cmd
                    # For convenient, we replace "M" and "BH" in the command to represent NumPy or Bohrium
                    cmd_np = cmd_np.replace("M", "np").replace("BH", "False")
                    cmd_bh = cmd_bh.replace("M", "bh").replace("BH", "True")
                    if args.verbose:
                        print("%s  [BH CMD] %s%s" % (OKBLUE, cmd_bh, ENDC))

                    # Let's execute the two commands
                    env = {"np": numpy, "bh": bohrium}
                    exec(cmd_np, env)
                    res_np = env['res']
                    env = {"np": numpy, "bh": bohrium}
                    exec (cmd_bh, env)
                    res_bh = env['res'].copy2numpy()
                    if not numpy.allclose(res_np, res_bh, equal_nan=True):
                        print("%s  [Error] %s%s" % (FAIL, name, ENDC))
                        print("%s  [BH CMD] %s%s" % (OKBLUE, cmd_np, ENDC))
                        print("%s  [BH RES]\n%s%s" % (OKGREEN, res_np, ENDC))
                        print("%s  [BH CMD] %s%s" % (OKBLUE, cmd_bh, ENDC))
                        print("%s  [BH RES]\n%s%s" % (FAIL, res_bh, ENDC))
                        if not args.cont_on_error:
                            sys.exit(1)
                print("%s (%.2fs) ✓%s" % (OKBLUE, time.time() - start_time, ENDC))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs the test suite')
    parser.add_argument(
        'files',
        type=str,
        nargs='+',
        help='The test files to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print test CMD"
    )
    parser.add_argument(
        '--cont-on-error',
        action='store_true',
        help="Continue on failed tests"
    )
    args = parser.parse_args()

    time_start = time.time()
    run(args)
    print("*** Finished in: %s%.2fs%s" % (OKBLUE, time.time() - time_start, ENDC))
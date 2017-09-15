# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import sys
import os
import imp

import numpy
import bohrium

# basestring is not available in Python 3
try:
  basestring
except NameError:
  basestring = str

# Never run test with the '-m bohrium' switch
assert (numpy != bohrium)

# Terminal colors
HEADER  = '\033[35m'
OKBLUE  = '\033[34m'
OKGREEN = '\033[32m'
WARNING = '\033[33m'
FAIL    = '\033[31m'
ENDC    = '\033[0m'


def get_test_object_names(obj):
    """ Returns all attribute names that starts with "test_" """
    ret = []

    for o in dir(obj):
        if o.startswith("test_"):
            ret.append(o)

    return ret


def check_result(res_np, res_bh):
    if isinstance(res_np, type):
        return res_np is res_bh
    if isinstance(res_np, basestring):
        return res_np == res_bh

    if bohrium.is_scalar(res_np):
        if not bohrium.is_scalar(res_bh):
            return False
    elif res_bh.size == 0 and res_bh.size == 0:
        return True  # Empty arrays are considered equal
    elif res_bh.shape != res_np.shape:
        return False
    try:
        return numpy.allclose(res_np, res_bh, equal_nan=True)
    except TypeError:
        # Old versions of NumPy do not have the 'equal_nan' option
        return numpy.allclose(res_np, res_bh)


def run(args):
    for filename in args.files:
        if not filename.endswith("py"):
            # Ignore non-python files
            continue

        # Remove ".py"
        module_name = os.path.basename(filename)[:-3]
        m = imp.load_source(module_name, filename)

        if len(args.class_list) > 0:
            cls_name_list = args.class_list
        else:
            cls_name_list = get_test_object_names(m)

        for cls_name in cls_name_list:
            if cls_name in args.exclude_class:
                continue

            cls_obj = getattr(m, cls_name)
            cls_inst = cls_obj()

            for mth_name in get_test_object_names(cls_obj):
                mth_obj = getattr(cls_inst, mth_name)
                name = "%s/%s/%s" % (filename, cls_name[5:], mth_name[5:])

                print("Testing %s%s%s " % (OKGREEN, name, ENDC), end="")
                sys.stdout.flush()

                start_time = time.time()

                for ret in getattr(cls_inst, "init")():
                    # Let's retrieve the NumPy and Bohrium commands
                    cmd = mth_obj(ret)
                    if len(cmd) == 2:
                        (cmd_np, cmd_bh) = cmd
                    else:
                        # If not returning a pair, the NumPy and Bohrium command are identical
                        cmd_np = cmd
                        cmd_bh = cmd

                    # For convenient, we replace "M" and "BH" in the command to represent NumPy or Bohrium
                    cmd_np = cmd_np.replace("M", "np").replace("BH", "False")
                    cmd_bh = cmd_bh.replace("M", "bh").replace("BH", "True")
                    if args.verbose:
                        print("%s  [NP CMD] %s%s" % (OKBLUE, cmd_np, ENDC))
                        print("%s  [BH CMD] %s%s" % (OKBLUE, cmd_bh, ENDC))

                    # Let's execute the two commands
                    env = {"np": numpy, "bh": bohrium}
                    exec (cmd_np, env)
                    res_np = env['res']
                    if bohrium.check(res_np):
                        print("\n")
                        print("%s  [Error]  The NumPy command returns a Bohrium array!%s" % (FAIL, ENDC))
                        print("%s  [NP CMD] %s%s" % (OKBLUE, cmd_np, ENDC))
                        print("%s  [NP RES]\n%s%s" % (OKGREEN, res_np, ENDC))
                        if not args.cont_on_error:
                            sys.exit(1)

                    env = {"np": numpy, "bh": bohrium}
                    exec (cmd_bh, env)

                    if bohrium.check(env['res']):
                        res_bh = env['res'].copy2numpy()
                    else:
                        res_bh = env['res']

                    if not check_result(res_np, res_bh):
                        print("\n")
                        print("%s  [Error]  %s%s" % (FAIL, name, ENDC))
                        print("%s  [NP CMD] %s%s" % (OKBLUE, cmd_np, ENDC))
                        print("%s  [NP RES]\n%s%s" % (OKGREEN, res_np, ENDC))
                        print("%s  [BH CMD] %s%s" % (OKBLUE, cmd_bh, ENDC))
                        print("%s  [BH RES]\n%s%s" % (FAIL, res_bh, ENDC))

                        if not args.cont_on_error:
                            sys.exit(1)

                print("%s(%.2fs) %s✓%s" % (OKBLUE, time.time() - start_time, OKGREEN, ENDC))


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

    parser.add_argument(
        '--class',
        type=str,
        action='append',
        default=[],
        help="Choose specific test class (the prefix 'test_' is ignored) " \
             "(supports multiple use of this argument)"
    )

    parser.add_argument(
        '--exclude-class',
        type=str,
        action='append',
        default=[],
        metavar='CLASS',
        help="Ignore specific test class (the prefix 'test_' is ignored) " \
             "(supports multiple use of this argument)"
    )
    args = parser.parse_args()

    # We need to rename class since it's a Python keyword
    args.class_list = getattr(args, "class")
    delattr(args, "class")

    # And make sure that all class names starts with "test_"
    for i in range(len(args.class_list)):
        if not args.class_list[i].startswith("test_"):
            args.class_list[i] = "test_%s" % args.class_list[i]

    for i in range(len(args.exclude_class)):
        if not args.exclude_class[i].startswith("test_"):
            args.exclude_class[i] = "test_%s" % args.exclude_class[i]

    time_start = time.time()
    run(args)

    print("%s***%s Finished in: %s%.2fs%s %s***%s" % (OKGREEN, ENDC, OKBLUE, time.time() - time_start, ENDC, OKGREEN, ENDC))

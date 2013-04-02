#!/usr/bin/env python
import subprocess
import argparse
import pprint
import sys
import re
import os

def count(fn):
    with open(fn) as fd:
        pass
    stats = {}
    return stats

def main():

    p = argparse.ArgumentParser(description='Counts FLoating-point operations.')
    p.add_argument(
        'filename',
        help='Path / filename of the trace-file'
    )
    p.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help="List of opcodes to exclude from parsing.\nExample: FREE,DISCARD,SYNC"
    )

    args = p.parse_args()

    if not os.path.exists(args.filename) or not os.path.isfile(args.filename):
        return "Error: invalid filename <%s>." % args.filename

    tracefile = args.filename
    
    # Count
    return pprint.pformat(count(tracefile)), ""

    
if __name__ == "__main__":
    out, err = main()
    if err:
        print "Error: %s" % err
    if out:
        print "Stats: %s" % out


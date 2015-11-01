#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):

    with open(args.output, 'w') as outfile:
        for fname in args.files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = "Concatenate files"
    )
    parser.add_argument(
        'files',
        default=[],
        nargs='*',
        help='Files to concatenate.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file.'
    )
    args = parser.parse_args()
    if args.output is None:
        raise ValueError("No --output was given!")
    main(args)


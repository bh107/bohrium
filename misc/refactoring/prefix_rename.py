#!/usr/bin/env python
import argparse
import shutil
import glob
import sys
import os

def rename( prefix, sep, text, directory='.', recursive=False, verbose=False, dry=True ):

    for root, dirs, files in os.walk( directory, topdown=True ):

        if not recursive and root != directory:
            break

        for fn in files:
            parts = fn.split(sep)

            if prefix == parts[0]:
                src = root +os.sep+ fn
                dst = root +os.sep+ fn.replace('%s%s' % (prefix, sep), '%s%s' % (text, sep), 1)

                if verbose or dry:
                    print "%s -> %s" % (src, dst)
                if not dry:
                    shutil.move(src, dst) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Renames / replaces a prefix of filenames.')
    parser.add_argument(
        'prefix',
        help="The prefix to replace"
    )
    parser.add_argument(
        'text',
        help="The string to replace the prefix with."
    )
    parser.add_argument(
        '-s',
        default="_",
        help="Seperater between the prefix and the remainder of the filename"
    )
    parser.add_argument(
        '-d',
        default=".",
        help="Dir in which to rename files."
    )
    parser.add_argument(
        '-r',
        default=False,
        action='store_true',
        help="Recursively scan for files to rename."
    )
    parser.add_argument(
        '-v',
        default=False,
        action='store_true',
        help="Show the files that will be renamed."
    )
    parser.add_argument(
        '-vv',
        default=False,
        action='store_true',
        help="Dry run: ONLY show the renaming, do not actually rename."
    )
    args = parser.parse_args()

    rename(args.prefix, args.s, args.text, args.d, args.r, args.v, args.vv)


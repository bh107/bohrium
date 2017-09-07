#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
"""

import argparse
import sys
import subprocess
import re
import os
import io


def c89fy(src):
    print "*"*1000
    print src
    print "*" * 100

    # Find all variable declarations
    symbols = set()
    ret = src[:]
    for m in re.finditer("\w+ ([its]\d+)", src):
        ret = ret.replace(m.group(0), m.group(1))
        symbols.add(m.group(0))
    sym_decl = ""
    for s in symbols:
        sym_decl += "    %s;\n" % s

    # Insert variable declarations and #pragma scop at the of the execute() function
    exec_func = "%s\n" % re.search("void execute\(.*\).*", ret).group(0)
    ret = ret.replace(exec_func, "%s%s\n#pragma scop\n" % (exec_func, sym_decl))

    # Replace Peeled loop with 1-sized for loops
    peeled_loop = "{ // Peeled loop, 1. sweep iteration\s*(i\d) = 0;"
    m = re.search(peeled_loop, ret, re.MULTILINE)
    if m is not None:
        ret = ret.replace(m.group(0), "for(%s=0; %s<1; ++%s){ // Peeled loop\n" % (m.group(1), m.group(1), m.group(1)))

    # Insert #pragma endscop at the end of the execute() function
    ret = ret.replace("\n}\n\nvoid launcher", "\n#pragma endscop\n}\n\n\nvoid launcher")

    print ret
    print symbols
    return ret


def main(args):
    print ("template: '%s'" % args.template)

    tmp_name = "%s.tmp.c" % args.src.name
    print "tmp_name: %s" % tmp_name
    with open(tmp_name, "w") as f:
        f.write(c89fy(args.src.read()))
        f.flush()
        os.fsync(f.fileno())

    polly_name = "%s.polly.c" % tmp_name
    cmd_polly = "%s %s -o %s" % (args.polycc, tmp_name, polly_name)
    print ("cmd_polly: %s" % cmd_polly)
    print subprocess.check_output(cmd_polly, shell=True)

    polly_name = tmp_name

    cmd_cc = args.template.replace("{WOUT}", args.dst).replace("{WIN}", polly_name)
    print ("cmd_cc: %s" % cmd_cc)
    print subprocess.check_output(cmd_cc, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pluto Compiler Wrapper')
    parser.add_argument(
        'template',
        type=str,
    )
    parser.add_argument(
        'src',
        type=argparse.FileType('r')
    )
    parser.add_argument(
        'dst',
        type=str
    )
    parser.add_argument(
        '--polycc',
        type=str,
        default='polycc'
    )
    args = parser.parse_args()
    main(args)

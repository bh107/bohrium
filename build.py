#!/usr/bin/python
"""
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of CphVB <https://github.com/cphvb>.
 *
 * DistNumPy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DistNumPy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DistNumPy. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
from os.path import join
import getopt
import subprocess

def build(name, dir, fatal=False):
    print "***Building %s***"%name
    try:
        p = subprocess.Popen(["make"], cwd=join(install_dir, dir))
        err = p.wait()
    except KeyboardInterrupt:
        p.terminate()

    if fatal:
        if err != 0:
            print "An build error in %s is fatal. Exiting."%name
            sys.exit(-1)
    else:
        if err != 0:
            print "An build error in %s is not fatal. Continuing."%name

def clean(name, dir):
    print "***Cleaning %s***"%name
    try:
        p = subprocess.Popen(["make", "clean"], cwd=join(install_dir, dir))
        err = p.wait()
    except KeyboardInterrupt:
        p.terminate()

def install(name, dir, fatal=False):
    print "***Installing %s***"%name
    try:
        p = subprocess.Popen(["make", "install"], cwd=join(install_dir, dir))
        err = p.wait()
    except KeyboardInterrupt:
        p.terminate()

    if fatal:
        if err != 0:
            print "An build error in %s is fatal. Exiting."%name
            sys.exit(-1)
    else:
        if err != 0:
            print "An build error in %s is not fatal. Continuing."%name

def ldconfig():
    print "***Configure ldconfig***"
    print "sudo ldconfig"
    try:
        p = subprocess.Popen(["sudo", "ldconfig"], cwd=join(install_dir))
        err = p.wait()
    except KeyboardInterrupt:
        p.terminate()

if __name__ == "__main__":
    debug = False
    prefix = "/usr/local/lib"
    try:
        install_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        print "The build script cannot run interactively."
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:],"d",["debug"])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ("-d", "--debug"):
            debug = True
        else:
            assert False, "unhandled option"
    try:
        cmd = args[0]
    except IndexError:
        print "No command given"
        sys.exit(-1)

    if cmd == "build":
        build("INIPARSER", "iniparser", True)
        build("CORE", "core", True)
        build("VE-CUDA", "ve/cuda", False)
        build("VE-SIMPLE", "ve/simple", False)
        build("VE-SCORE", "ve/score", True)
        build("VE-OMP", "ve/omp", False)
        build("VEM-NODE", "vem/node", True)
        build("VEM-CLUSTER", "vem/cluster", False)
        build("BRIDGE-NUMPY", "bridge/numpy", True)
    elif cmd == "clean":
        clean("INIPARSER", "iniparser")
        clean("CORE", "core")
        clean("VE-CUDA", "ve/cuda")
        clean("VE-SIMPLE", "ve/simple")
        clean("VE-SCORE", "ve/score")
        clean("VE-OMP", "ve/omp")
        clean("VEM-NODE", "vem/node")
        clean("VEM-CLUSTER", "vem/cluster")
        clean("BRIDGE-NUMPY", "bridge/numpy")
    elif cmd == "install":
        install("INIPARSER", "iniparser", True)
        install("CORE", "core", True)
        install("VE-CUDA", "ve/cuda", False)
        install("VE-SIMPLE", "ve/simple", True)
        install("VE-SCORE", "ve/score",True)
        install("VE-OMP", "ve/omp",False)
        install("VEM-NODE", "vem/node", True)
        install("VEM-CLUSTER", "vem/cluster", False)
        install("BRIDGE-NUMPY", "bridge/numpy",True)
        ldconfig()
    else:
        print "Unknown command: '%s'."%cmd

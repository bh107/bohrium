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
from os.path import join, expanduser, exists
import shutil
import getopt
import subprocess

makecommand = "make"
makefilename = "Makefile"

def build(components):
    for (name, dir, fatal) in components:
        print "***Building %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename], cwd=join(install_dir, dir))
            err = p.wait()
        except KeyboardInterrupt:
            p.terminate()

        if fatal:
            if err != 0:
                print "A build error in %s is fatal. Exiting."%name
                sys.exit(-1)
        else:
            if err != 0:
                print "A build error in %s is not fatal. Continuing."%name

def clean(components):
    for (name, dir, fatal) in components:
        print "***Cleaning %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename, "clean"], cwd=join(install_dir, dir))
            err = p.wait()
        except KeyboardInterrupt:
            p.terminate()

def install(components):
    for (name, dir, fatal) in components:
        print "***Installing %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename, "install"], cwd=join(install_dir, dir))
            err = p.wait()
        except KeyboardInterrupt:
            p.terminate()

        if fatal:
            if err != 0:
                print "A build error in %s is fatal. Exiting."%name
                sys.exit(-1)
        else:
            if err != 0:
                print "A build error in %s is not fatal. Continuing."%name

def ldconfig():
    print "***Configure ldconfig***"
    print "sudo ldconfig"
    try:
        p = subprocess.Popen(["sudo", "ldconfig"], cwd=join(install_dir))
        err = p.wait()
    except KeyboardInterrupt:
        p.terminate()

def install_config():
    HOME_CONFIG = join(join(expanduser("~"),".cphvb"))
    if not exists(HOME_CONFIG):
        os.mkdir(HOME_CONFIG)
        dst = join(HOME_CONFIG, "config.ini")
        src = join(install_dir,"config.ini.example")
        shutil.copy(src,dst)
        print "cp %s %s"%(src,dst)


if __name__ == "__main__":
    debug = False
    prefix = "/opt/cphvb"
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

    if sys.platform.startswith('win32'):
        makecommand="nmake"
        makefilename="Makefile.win"
    elif sys.platform.startswith('darwin'):
        makefilename="Makefile.osx"

    try:
        cmd = args[0]
    except IndexError:
        print "No command given"
        print ""
        print "Known commands: build, clean, install, rebuild"
        sys.exit(-1)

    components = [\
                  ("OPCODES","core/codegen",True),\
                  ("INIPARSER","iniparser",True),\
                  ("CORE-BUNDLER", "core/bundler", True),\
                  ("CORE-COMPUTE", "core/compute", True),\
                  ("CORE", "core", True),\
                  ("VE-GPU", "ve/gpu", False),\
                  ("VE-SIMPLE", "ve/simple", True),\
                  ("VE-SCORE", "ve/score", False),\
                  ("VE-MCORE", "ve/mcore", False),\
                  ("VEM-NODE", "vem/node", True),\
                  ("BRIDGE-NUMPY", "bridge/numpy", True),\
                  ("USERFUNCS-ATLAS", "userfuncs/ATLAS", False),\
                  ("CPHVBNUMPY", "cphvbnumpy", True),\
                 ]

    if cmd == "rebuild":
        clean(components)        
    if cmd == "build" or cmd == "rebuild":
        build(components)        
    elif cmd == "clean":
        clean(components)        
    elif cmd == "install":
        if not exists("/opt/cphvb"):
            os.mkdir("/opt/cphvb")
        install(components)        
        install_config();
    else:
        print "Unknown command: '%s'."%cmd
        print ""
        print "Known commands: build, clean, install"

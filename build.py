#!/usr/bin/python
"""
/*
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
*/
"""

import sys
import os
from os.path import join, expanduser, exists
import shutil
import getopt
import subprocess

makecommand = "make"
makefilename = "Makefile"

def build(components,interpreter):
    for (name, dir, fatal) in components:
        print "***Building %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename,"BH_PYTHON=%s"%interpreter], cwd=join(install_dir, dir))
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

def install(components,prefix,interpreter):
    if not exists(join(prefix,"lib")):
        os.mkdir(join(prefix,"lib"))
    for (name, dir, fatal) in components:
        print "***Installing %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename,"install","BH_PYTHON=%s"%interpreter,"INSTALLDIR=%s"%prefix], cwd=join(install_dir, dir))
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

def install_config(prefix):
    if os.geteuid() == 0:#Root user
        HOME_CONFIG = "/etc/bohrium"
    else:
        HOME_CONFIG = join(join(expanduser("~"),".bohrium"))
    if not exists(HOME_CONFIG):
        os.mkdir(HOME_CONFIG)
    dst = join(HOME_CONFIG, "config.ini")
    src = join(install_dir,"config.ini.example")
    if not exists(dst):
        src_file = open(src, "r")
        src_str = src_file.read()
        src_file.close()
        dst_str = src_str.replace("/opt/bohrium",prefix)
        if sys.platform.startswith('darwin'):
            dst_str = dst_str.replace(".so",".dylib")
        dst_file = open(dst,"w")
        dst_file.write(dst_str)
        dst_file.close()
        print "Write default config file to %s"%(dst)


if __name__ == "__main__":
    debug = False
    interactive = False
    if not sys.platform.startswith('win32') and os.geteuid() == 0:#Root user
        prefix = "/opt/bohrium"
    else:
        prefix = join(join(expanduser("~"),".local"))
    interpreter = sys.executable
    try:
        install_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        print "The build script cannot run interactively."
        sys.exit(-1)

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:],"d",["debug","prefix=","interactive","interpreter="])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ("-d","--debug"):
            debug = True
        elif o in ("--prefix"):
            prefix = a
        elif o in ("--interactive"):
            interactive = True
        elif o in ("--interpreter"):
            interpreter = a
        else:
            assert False, "unhandled option"

    if sys.platform.startswith('win32'):
        makecommand="nmake"
        makefilename="Makefile.win"
    elif sys.platform.startswith('darwin'):
        makefilename="Makefile.osx"

    if interactive:
        import readline, glob
        def complete(text, state):#For autocomplete
            return (glob.glob(text+'*')+[None])[state]
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(complete)

        print "Please specify the installation directory:"
        answer = raw_input("[%s] "%prefix)
        if answer != "":
            prefix = expanduser(answer)
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
                  ("CORE-BHIR", "core/bhir", True),\
                  ("CORE", "core", True),\
                  ("VE-SHARED-COMPUTE", "ve/shared/compute", True),\
                  ("VE-SHARED-BUNDLER", "ve/shared/bundler", False),\
                  #("VE-GPU", "ve/gpu", False),\
                  ("VE-CPU",    "ve/cpu", True),\
                  ("VE-SCORE",  "ve/static/score", False),\
                  ("VE-MCORE",  "ve/static/mcore", False),\
                  ("VE-TILING", "ve/static/tiling", False),\
                  ("VEM-NODE", "vem/node", True),\
                  ("VEM-CLUSTER", "vem/cluster", False),\
                  #("FILTER-POWER", "filter/power", False),\
                  #("FILTER-FUSION", "filter/fusion", False),\
                  #("FILTER-STREAMING", "filter/streaming", False),\
                  ("FILTER-PPRINT", "filter/pprint", True),\
                  ("FILTER-TRANSITIVE-REDUCTION", "filter/transitive_reduction", True),\
                  #("NumCIL", "bridge/NumCIL", False),\
                  ("BRIDGE-NUMPY", "bridge/numpy", True),\
                  #("USERFUNCS-ATLAS", "userfuncs/ATLAS", False),\
                  ("BHNUMPY", "bohrium", True)
                 ]

    if cmd == "rebuild":
        clean(components)
    if cmd == "build" or cmd == "rebuild":
        build(components,interpreter)
    elif cmd == "clean":
        clean(components)
    elif cmd == "install":
        prefix = os.path.abspath(prefix)
        if exists(prefix):
            assert os.path.isdir(prefix),"The prefix points to an existing file"
        else:
            os.makedirs(prefix)
        install(components,prefix,interpreter)
        install_config(prefix);
    else:
        print "Unknown command: '%s'."%cmd
        print ""
        print "Known commands: build, clean, install"

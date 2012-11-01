#!/usr/bin/python
"""
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

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
            p = subprocess.Popen([makecommand, "-f", makefilename,"CPHVB_PYTHON=%s"%interpreter], cwd=join(install_dir, dir))
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
    for (name, dir, fatal) in components:
        print "***Installing %s***"%name
        try:
            p = subprocess.Popen([makecommand, "-f", makefilename,"install","CPHVB_PYTHON=%s"%interpreter,"INSTALLDIR=%s"%prefix], cwd=join(install_dir, dir))
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
        HOME_CONFIG = "/etc/cphvb"
    else: 
        HOME_CONFIG = join(join(expanduser("~"),".cphvb"))
    if not exists(HOME_CONFIG):
        os.mkdir(HOME_CONFIG)
    dst = join(HOME_CONFIG, "config.ini")
    src = join(install_dir,"config.ini.example")
    if not exists(dst):
        src_file = open(src, "r")
        src_str = src_file.read()
        src_file.close()
        dst_str = src_str.replace("/opt/cphvb",prefix)
        if sys.platform.startswith('darwin'):
            dst_str = dst_str.replace(".so",".dylib")
        dst_file = open(dst,"w")
        dst_file.write(dst_str)
        dst_file.close()
        print "Write default config file to %s"%(dst)


if __name__ == "__main__":
    debug = False
    interactive = False
    prefix = "/opt/cphvb"
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
                  ("CORE-BUNDLER", "core/bundler", True),\
                  ("CORE-COMPUTE", "core/compute", True),\
                  ("CORE", "core", True),\
                  ("VE-GPU", "ve/gpu", False),\
                  ("VE-SIMPLE", "ve/simple", True),\
                  ("VE-TILE", "ve/tile", False),\
                  ("VE-NAIVE", "ve/naive", False),\
                  ("VE-SCORE", "ve/score", False),\
                  ("VE-MCORE", "ve/mcore", False),\
                  ("VEM-NODE", "vem/node", True),\
                  ("VEM-CLUSTER", "vem/cluster", False),\
                  ("NumCIL", "bridge/NumCIL", False),\
                  ("BRIDGE-NUMPY", "bridge/numpy", True),\
                  ("USERFUNCS-ATLAS", "userfuncs/ATLAS", False),\
                  ("CPHVBNUMPY", "cphvbnumpy", True)
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

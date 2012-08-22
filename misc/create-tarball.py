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

import os
import sys
import subprocess


if __name__ == "__main__":
    try:
        (root,_) = os.path.split(os.path.abspath(os.path.dirname(__file__)))
    except NameError:
        print "This script cannot run interactively."
        sys.exit(-1)

    VERSION = "v0.1"
    NAME = "cphvb-%s"%VERSION

    try:
        p = subprocess.Popen(["git", "archive", "--format=tar", "--prefix=cphvb-dump/", "-o", "/tmp/cphvb-root-dump.tar", "HEAD"], cwd=root)
        p.wait()
        p = subprocess.Popen(["git", "archive", "--format=tar", "--prefix=cphvb-numpy-dump/", "-o", "/tmp/cphvb-numpy-dump.tar", "HEAD"], cwd=os.path.join(root,"bridge","numpy"))
        p.wait()
        p = subprocess.Popen(["tar", "-xf", "cphvb-root-dump.tar"], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["tar", "-xf", "cphvb-numpy-dump.tar"], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["rmdir", "cphvb-dump/bridge/numpy"], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["mv", "cphvb-numpy-dump","cphvb-dump/bridge/numpy"], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["mv", "cphvb-dump",NAME], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["tar", "-czf", "%s.tgz"%NAME, NAME], cwd="/tmp")
        p.wait()
        p = subprocess.Popen(["rm", "-R", "cphvb-root-dump.tar", "cphvb-numpy-dump.tar", NAME], cwd="/tmp")
        p.wait()
    except KeyboardInterrupt:
        p.terminate()

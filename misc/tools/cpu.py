#!/usr/bin/env python
#
# This file is part of Bohrium and copyright (c) 2013 the Bohrium team:
# http://cphvb.bitbucket.org
#
# Bohrium is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 
# of the License, or (at your option) any later version.
#
# Bohrium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the 
# GNU Lesser General Public License along with Bohrium. 
#
# If not, see <http://www.gnu.org/licenses/>.
#
#!/usr/bin/env python
import subprocess
import glob
import time
import os
import re

import bhutils

def merge_kernels(config):
    """
    Creates a shared library named 'bh_libsij.so' containing
    all the functions defined in the 'kernel_path'.
    A 'bh_libsij.idx' is also produced containing all the bohrium
    functions in the '.so' file.
    """

    times = [('start', time.time())]

    krn_path = config.get('cpu', 'kernel_path')
    obj_path = config.get('cpu', 'object_path')
    idx_path = "%s%s%s" % (obj_path, os.sep, "LIB_libsij_aaaaaa.idx")
    lib_path = "%s%s%s" % (obj_path, os.sep, "LIB_libsij_aaaaaa.so")

    if not os.path.exists(krn_path):
        return (None, "kernel_path(%s) does not exist." % krn_path)

    if not os.path.exists(obj_path):
        return (None, "obj_path(%s) does not exist." % obj_path)

    cmd = [c for c in config.get('cpu',
                                 'compiler_cmd').replace('"','').split(' ')
            if c] + [lib_path]

    symbols = []                                # Find the source-files
    sources = []
    files   = []
    for fn in glob.glob("%s%sKRN_*.c" % (krn_path, os.sep)):
        m = re.match('.*KRN_(\d+)_([a-zA-Z0-9]{6}).c', fn)
        if m:
            symbol, instance = m.groups()
            if symbol not in symbols:           # Ignore duplicates
                sources.append(open(fn, 'r').read())
                symbols.append(symbol)
                files.append(fn)
    
    source = "\n".join(sources)                 # Compile them
    times.append(('merged', time.time()))

    p = subprocess.Popen(
        cmd,
        stdin   = subprocess.PIPE,
        stdout  = subprocess.PIPE
    )
    out, err = p.communicate(input=source)
    times.append(('compiled', time.time()))

    with open(idx_path, 'w+') as fd:            # Create the index-file
        symbols.sort()
        fd.write("\n".join(symbols))

    times.append(('done', time.time()))
    bhutils.print_timings(times)

    return (out, err)


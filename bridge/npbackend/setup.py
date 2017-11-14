#!/usr/bin/env python
"""
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
http://bohrium.bitbucket.org

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

from distutils.core import setup, Extension
from distutils.command.build import build
import os
import sys
import stat
import pprint
import json
import shutil
import argparse
import numpy as np
from Cython.Distutils import build_ext

# We overload the setup.py with some extra arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--buildpath',
    help='Path to the build directory.'
)
parser.add_argument(
    '--openmp-flag',
    default="",
    help='OpenMP flag for the Cython builds'
)
(args_extra, argv) = parser.parse_known_args()
sys.argv = [sys.argv[0]] + argv  # Write the remaining arguments back to `sys.argv` for distutils to read


def buildpath(*paths):
    if args_extra.buildpath is None:
        return os.path.join(*paths)
    else:
        return os.path.join(args_extra.buildpath, *paths)


def srcpath(*paths):
    prefix = os.path.abspath(os.path.dirname(__file__))
    assert len(prefix) > 0
    return os.path.join(prefix, *paths)


def get_timestamp(f):
    st = os.stat(f)
    mtime = st[stat.ST_MTIME]  # modification time
    return mtime


def set_timestamp(f, timestamp):
    os.utime(f, (timestamp, timestamp))


# Returns the numpy data type name
def dtype_bh2np(bh_type_str):
    return bh_type_str[3:].lower()  # Remove BH_ and convert to lower case


# Merge bhc.i.head with the bh_c.h to create our SWIG interface bhc.i
time = 0
with open(buildpath("bhc.i"), 'w') as outfile:
    for fname in [srcpath("bhc.i.head"), buildpath("..", "c", "out", "bhc.h")]:
        t = get_timestamp(fname)
        if t > time:
            time = t
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
set_timestamp(buildpath("bhc.i"), time)

# Information variables that should be written to the _info.py file
info_vars = {}

# The version if written in the VERSION file in the root of Bohrium
_version = "0.0.0"
with open(srcpath("..", "..", "VERSION"), "r") as f:
    _version = f.read().strip()
    info_vars['__version__'] = _version

# NumPy info
info_vars['version_numpy'] = np.__version__

# Create the _info.py file
time = get_timestamp(srcpath('setup.py'))
with open(buildpath("_info.py"), 'w') as o:
    # Write header
    o.write("#This file is auto generated by the setup.py\n")
    o.write("import numpy as np\n")

    # Write the information variables
    o.write("\n# Info variables:\n")
    for (key, val) in info_vars.items():
        o.write("%s = '%s'\n" % (key, val))
    o.write("\n")

    # Find number of operands and type signature for each Bohrium opcode
    # that Bohrium-C supports
    t = get_timestamp(srcpath('..', '..', 'core', 'codegen', 'opcodes.json'))
    if t > time:
        time = t
    nops = {}
    type_sig = {}

    ufunc = {}
    with open(srcpath('..', '..', 'core', 'codegen', 'opcodes.json'), 'r') as f:
        opcodes = json.loads(f.read())
        for op in opcodes:
            if op['elementwise'] and not op['system_opcode']:
                # Convert the type signature to bhc names
                type_sig = []
                for sig in op['types']:
                    type_sig.append([dtype_bh2np(s) for s in sig])

                name = op['opcode'].lower()[3:]  # Removing BH_ and we have the NumPy and bohrium name
                ufunc[name] = {'name': name,
                               'nop': int(op['nop']),
                               'type_sig': type_sig}
    o.write("op = ")
    pp = pprint.PrettyPrinter(indent=2, stream=o)
    pp.pprint(ufunc)

    # Find and write all supported data types
    t = get_timestamp(srcpath('..', '..', 'core', 'codegen', 'types.json'))
    if t > time:
        time = t
    s = "numpy_types = ["
    with open(srcpath('..', '..', 'core', 'codegen', 'types.json'), 'r') as f:
        types = json.loads(f.read())
        for t in types:
            if t['numpy'] == "unknown":
                continue
            s += "np.dtype('%s'), " % t['numpy']
        s = s[:-2] + "]\n"
    o.write(s)
set_timestamp(buildpath("_info.py"), time)


# We need to make sure that the extensions is build before the python module because of SWIG
# Furthermore, '_info.py' and 'bhc.py' should be copied to the build dir
class CustomBuild(build):
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]

    def run(self):
        if not self.dry_run:
            self.copy_file(buildpath('_info.py'), buildpath(self.build_lib, 'bohrium', '_info.py'))
            self.copy_file(buildpath('bhc.py'), buildpath(self.build_lib, 'bohrium', 'bhc.py'))
        build.run(self)

# We need the pyx files in the build path for the Cython.Distutils to work
shutil.copy2(srcpath('bohrium', 'random123.pyx'), buildpath('random123.pyx'))
shutil.copy2(srcpath('bohrium', 'bhary.pyx'), buildpath('bhary.pyx'))
shutil.copy2(srcpath('bohrium', '_util.pyx'), buildpath('_util.pyx'))
shutil.copy2(srcpath('bohrium', 'ufuncs.pyx'), buildpath('ufuncs.pyx'))
try:
    os.makedirs(buildpath('nobh'))
except OSError:
    pass
shutil.copy2(srcpath('bohrium', 'nobh', 'bincount_cython.pyx'), buildpath('nobh', 'bincount_cython.pyx'))

setup(name='Bohrium',
      version=_version,
      description='Bohrium NumPy',
      long_description='Bohrium NumPy',
      author='The Bohrium Team',
      author_email='contact@bh107.org',
      url='http://www.bh107.org',
      license='LGPLv3',
      platforms='Linux, OSX',
      cmdclass={'build': CustomBuild, 'build_ext': build_ext},
      package_dir={'bohrium': srcpath('bohrium')},
      packages=['bohrium', 'bohrium.target', 'bohrium.nobh'],
      ext_package='bohrium',
      ext_modules=[Extension(name='_bh',
                             sources=[srcpath('src', '_bh.cpp')],
                             depends=[srcpath('src', 'types.c'), srcpath('src', 'types.h'),
                                      srcpath('src', 'operator_overload.c')],
                             include_dirs=[buildpath("..", "c", "out"),
                                           srcpath('..', '..', 'include')],
                             libraries=['dl', 'bhc', 'bh'],
                             library_dirs=[buildpath('..', 'c'),
                                           buildpath('..', '..', 'core')],
                             ),
                   Extension(name='_bhc',
                             sources=[buildpath('bhc.i')],
                             include_dirs=[buildpath("..", "c", "out"),
                                           srcpath('..', '..', 'include')],
                             libraries=['dl', 'bhc', 'bh'],
                             library_dirs=[buildpath('..', 'c'),
                                           buildpath('..', '..', 'core')],
                             ),
                   Extension(name='random123', 
                             sources=[buildpath('random123.pyx')],
                             include_dirs=[srcpath('.'),
                                           srcpath('..', '..', 'thirdparty', 'Random123-1.09', 'include')],
                             libraries=[],
                             library_dirs=[],
                             ),
                   Extension(name='_util',
                             sources=[buildpath('_util.pyx')],
                             include_dirs=[srcpath('.')],
                             libraries=[],
                             library_dirs=[],
                             ),
                   Extension(name='bhary',
                             sources=[buildpath('bhary.pyx')],
                             include_dirs=[srcpath('.')],
                             libraries=[],
                             library_dirs=[],
                             ),
                   Extension(name='ufuncs',
                             sources=[buildpath('ufuncs.pyx')],
                             include_dirs=[srcpath('.')],
                             libraries=[],
                             library_dirs=[],
                             ),
                   Extension(name='nobh.bincount_cython',
                             sources=[buildpath("nobh", 'bincount_cython.pyx')],
                             include_dirs=[srcpath('.')],
                             libraries=[],
                             library_dirs=[],
                             extra_compile_args=[args_extra.openmp_flag],
                             extra_link_args=[args_extra.openmp_flag])
                   ]
      )

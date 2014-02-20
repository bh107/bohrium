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
from os.path import join

#Merge bhc.i.head with the bh_c.h to create our SWIG interface bhc.i
with open("bhc.i", 'w') as outfile:
    for fname in ["bhc.i.head","../c/codegen/output/bh_c.h"]:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

SRC  = ['_bhmodule.c']
DEPS = ['types.c', 'types.h']

setup(name='Bohrium',
      version='0.1',
      description='Bohrium NumPy',
      long_description='Bohrium NumPy',
      author='The Bohrium Team',
      author_email='contact@bh107.org',
      url='http://www.bh107.org',
      license='LGPLv3',
      platforms='Linux, OSX',

      packages=['bohrium'],
      ext_package='bohrium',
      ext_modules=[Extension(name='_bhmodule',
                             sources=[join('bohrium',f) for f in SRC],
                             depends=[join('bohrium',f) for f in DEPS],
                             include_dirs=[join('../c/codegen/output'),
                                           join('..','..','include')],
                             libraries=['dl','bhc', 'bh'],
                             library_dirs=[join('..','c'),
                                           join('..','..','core')],
                             ),
                   Extension(name='_bhc',
                             sources=['bhc.i'],
                             include_dirs=[join('../c/codegen/output'),
                                           join('..','..','include')],
                             libraries=['dl','bhc', 'bh'],
                             library_dirs=[join('..','c'),
                                           join('..','..','core')],
                             )],
     )

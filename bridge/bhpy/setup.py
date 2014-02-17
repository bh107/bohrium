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

setup(name='Bohrium Internal Interface',
      version='1.0',
      ext_modules=[Extension(name='_bhmodule',
                             sources=[join('src','_bhmodule.c')],
                             include_dirs=[join('../c/codegen/output'),
                                           join('..','..','include')],
                             libraries=['dl','bhc', 'bh'],
                             library_dirs=[join('..','c'),
                                           join('..','..','core')],
                             extra_compile_args=[],
#                             extra_link_args=['-L%s'%join(bohrium_install_dir,'core'), '-lbh'],
                             depends=[],
                             )],
     )

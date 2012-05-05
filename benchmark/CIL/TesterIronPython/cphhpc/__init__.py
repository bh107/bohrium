#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# --- BEGIN_HEADER ---
#
# __init__ - shared lib module init
# Copyright (C) 2011-2012  The cphhpc Project lead by Brian Vinter
#
# This file is part of CPHHPC Toolbox.
#
# CPHHPC Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CPHHPC Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
#
# -- END_HEADER ---
#

"""Copenhagen High Performance Computing Toolbox is a collection of
high performance libraries for flexible and efficient scientific computation
using Python.
"""


# All sub modules to load in case of 'from X import *'
__all__ = []


# Collect all package information here for easy use from escripts and helpers
package_name = "CPHHPC Toolbox"
short_name = "cphhpctoolbox"
version_tuple = (0, 0, 1)
version_string = ".".join([str(i) for i in version_tuple])
package_version = "%s %s" % (package_name, version_string)
project_team = "The CPHHPC project lead by Brian Vinter"
project_email = "brian DOT vinter AT gmail DOT com"
maintainer_team = "The CPHHPC Toolbox maintainers"
maintainer_email = "martin DOT rehr AT gmail DOT com"
project_url = "http://code.google.com/p/cphhpc/",
download_url = "http://code.google.com/p/cphhpc/downloads/list",
license_name = "GNU GPL v2",
short_desc = "CPHHPC Toolbox is a set of high performance Python extensions"
long_desc = """Copenhagen High Performance Computing Toolbox is a collection of
high performance libraries for flexible and efficient scientific computation
using Python.
"""

project_class = [
    'Development Status :: 1 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    ]
project_keywords = ['python', 'numpy', 'science', 'research', 'cpu', 'gpu']
# We can't really do anything useful without at least numpy
project_requires = ['numcil']

# Optional packages required for additional functionality (for extras_require)
project_extras = {}

package_provides = short_name
project_platforms = ['All']

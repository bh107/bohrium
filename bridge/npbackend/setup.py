#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
/*
This file is part of Bohrium and copyright (c) 2018 the Bohrium
<http://www.bh107.org>

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

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as setup_build_ext
import numbers
import os
import glob


def script_path(*paths):
    prefix = os.path.abspath(os.path.dirname(__file__))
    assert len(prefix) > 0
    return os.path.join(prefix, *paths)


def get_pyx_extensions():
    """Find and compiles all cython extensions"""
    include_dirs = []
    if 'USE_CYTHON' in os.environ:
        import numpy
        import bohrium_api
        include_dirs.extend([numpy.get_include(), bohrium_api.get_include()])
    pyx_list = glob.glob(script_path("bohrium", "*.pyx"))
    ret = []
    for pyx in pyx_list:
        ret.append(Extension(name="bohrium.%s" % os.path.splitext(os.path.basename(pyx))[0],
                             sources=[pyx],
                             include_dirs=include_dirs))
    ret.append(Extension(name="bohrium.nobh.bincount_cython",
                         sources=[script_path("bohrium", "nobh", "bincount_cython.pyx")],
                         include_dirs=include_dirs))
    if 'USE_CYTHON' in os.environ:
        import Cython.Build
        return Cython.Build.cythonize(ret, nthreads=2)
    else:
        return ret


class BuildExt(setup_build_ext):
    """We delay the numpy and bohrium dependency to the build command.
    Hopefully, PIP has installed them at this point."""

    def run(self):
        if not self.dry_run:
            import numpy
            import bohrium_api
            for ext in self.extensions:
                ext.include_dirs.extend([numpy.get_include(), bohrium_api.get_include()])
        setup_build_ext.run(self)


class DelayedVersion(numbers.Number):
    """In order to delay the version evaluation that depend on `bohrium_api`, we use this class"""

    def __str__(self):
        import bohrium_api
        return bohrium_api.__version__


setup(
    cmdclass={'build_ext': BuildExt},
    name='bohrium',
    version=DelayedVersion(),
    description='Bohrium Python/NumPy Backend',
    long_description='Bohrium for Python <www.bh107.org>',
    url='http://bh107.org',
    author='The Bohrium Team',
    author_email='contact@bh107.org',
    maintainer='Mads R. B. Kristensen',
    maintainer_email='madsbk@gmail.com',
    platforms=['Linux', 'OSX'],

    # Choose your license
    license='GPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
    ],

    # What does your project relate to?
    keywords='Bohrium, bh107, Python, C, HPC, MPI, PGAS, CUDA, OpenCL, OpenMP',

    # Dependencies
    install_requires=['numpy>=1.7', 'bohrium_api'],
    setup_requires=['numpy>=1.7', 'bohrium_api'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['tests']),

    ext_modules=get_pyx_extensions() + [
        Extension(
            name='bohrium._bh',
            sources=[script_path('src', '_bh.c'),
                     script_path('src', 'bharray.c'),
                     script_path('src', 'handle_array_op.c'),
                     script_path('src', 'handle_special_op.c'),
                     script_path('src', 'memory.c'),
                     script_path('src', 'util.c')],
            depends=[
                script_path('src', '_bh.h'),
                script_path('src', 'bharray.h'),
                script_path('src', 'util.h'),
                script_path('src', 'handle_array_op.h'),
                script_path('src', 'handle_special_op.h'),
                script_path('src', 'memory.h'),
                script_path('src', 'operator_overload.c')
            ],
            libraries=['dl'],
            extra_compile_args=["-std=c99"],
        )]
)

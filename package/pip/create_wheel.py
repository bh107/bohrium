#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Always prefer setuptools over distutils
from setuptools import setup
from distutils.dir_util import mkpath, copy_tree

# To use a consistent encoding
from codecs import open
import os
import re
import argparse
import sys
import shutil
import glob
import subprocess

from os.path import join

# We overload the setup.py with some extra arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--bh-install-prefix',
    help='Path to the Bohrium install prefix.'
)
parser.add_argument(
    '--npbackend-dir',
    help='Path to the Python library of Bohrium'
)
parser.add_argument(
    '--config',
    type=argparse.FileType('r'),
    help='Path to the Bohrium config file'
)
parser.add_argument(
    '-L',
    '--lib',
    type=str,
    action='append',
    help='Extra libraries to include in `lib64'
)
parser.add_argument(
    '-B',
    '--bin',
    type=str,
    action='append',
    help='Extra binaries to include in `bin`'
)
(args_extra, argv) = parser.parse_known_args()
sys.argv = [sys.argv[0]] + argv  # Write the remaining arguments back to `sys.argv` for distutils to read
assert(args_extra.bh_install_prefix)
assert(args_extra.npbackend_dir)
assert(args_extra.config)
assert("bdist_wheel" in sys.argv)

args_extra.bh_install_prefix = os.path.abspath(args_extra.bh_install_prefix)
args_extra.npbackend_dir = os.path.abspath(args_extra.npbackend_dir)


def _script_path():
    """Returns the path to the dir this script is in"""
    return os.path.dirname(os.path.realpath(__file__))


def _li1st_files(path, prefix=None, regex_include="\.so|\.ini|\.dylib"):
    """Returns a list of filenames of the files in `path`"""
    ret = []
    for f in os.listdir(path):
        if re.search(regex_include, f):
            if prefix is not None:
                f = prefix + f
            ret.append(f)
    return ret


def _find_data_files(root_path, regex_exclude=None, regex_include=None):
    """Return a list of paths relative to `root_path` of all files rooted in `root_path`"""
    root_path = os.path.abspath(root_path)
    ret = []
    for root, _, filenames in os.walk(root_path):
        for fname in filenames:
            fullname = join(root, fname)
            if regex_exclude is None or not re.search(regex_exclude, fullname):
                if regex_include is None or re.search(regex_include, fullname):
                    ret.append(fullname.replace("%s/" % root_path, ""))
    return ret


def _copy_files(glob_str, dst_dir):
    """Copy files using the `glob_str` and copy to `dst_dir`"""
    mkpath(dst_dir)
    for fname in glob.glob(glob_str):
        if os.path.isfile(fname):
            shutil.copy(fname, dst_dir)
            print("copy: %s => %s" % (fname, dst_dir))


def _copy_dirs(src_dir, dst_dir):
    """Copy dir"""
    mkpath(dst_dir)
    copy_tree(src_dir, dst_dir)
    print("copy %s => %s" % (src_dir, dst_dir))


def _regex_replace(pattern, repl, src):
    """Replacing matches in `src` with `repl` using regex `pattern`"""
    print ("config.ini: replacing: '%s' => '%s'" % (pattern, repl))
    return re.sub(pattern, repl, src)


# Copy the include dir for the JIT compilation into the Python package
_copy_dirs(join(args_extra.bh_install_prefix, "share", "bohrium", "include"),
           join(args_extra.npbackend_dir, "include"))


# Copy Python tests into the Python package
_copy_files(join(args_extra.bh_install_prefix, "share", "bohrium", "test", "python", "run.py"),
            join(args_extra.npbackend_dir, "test"))
_copy_files(join(args_extra.bh_install_prefix, "share", "bohrium", "test", "python", "util.py"),
            join(args_extra.npbackend_dir, "test"))
_copy_files(join(args_extra.bh_install_prefix, "share", "bohrium", "test", "python", "tests", "*.py"),
            join(args_extra.npbackend_dir, "test", "tests"))


# Copy Bohrium's shared libraries into the Python package
_copy_files(join(args_extra.bh_install_prefix, 'lib64', 'lib*'), join(args_extra.npbackend_dir, "lib64"))


# Copy extra libraries specified by the user
if args_extra.lib is not None:
    for lib in args_extra.lib:
        _copy_files(lib, join(args_extra.npbackend_dir, "lib64"))


# Copy extra libraries specified by the user
if args_extra.bin is not None:
    for bin in args_extra.bin:
        _copy_files(bin, join(args_extra.npbackend_dir, "bin"))


# Update the RPATH of the Python extensions to look in the the `lib64` dir
for filename in glob.glob(join(args_extra.npbackend_dir, '*.so')):
    cmd = "patchelf --set-rpath '$ORIGIN/lib64' %s" % filename
    print(cmd)
    subprocess.check_call(cmd, shell=True)


# Update the RPATH of Bohrium's shared libraries to look in the current dir
for filename in glob.glob(join(args_extra.npbackend_dir, 'lib64', '*')):
    cmd = "patchelf --set-rpath '$ORIGIN' %s" % filename
    print(cmd)
    subprocess.check_call(cmd, shell=True)


# Update the RPATH of Bohrium's binaries to look in the current dir
for filename in glob.glob(join(args_extra.npbackend_dir, 'bin', '*')):
    cmd = "patchelf --set-rpath '$ORIGIN/../lib64' %s" % filename
    print(cmd)
    subprocess.check_call(cmd, shell=True)


# Write a modified config file to the Python package dir
_config_path = join(args_extra.npbackend_dir, "config.ini")
_config_dir = os.path.dirname(_config_path)
with open(_config_path, "w") as f:
    config_str = args_extra.config.read()

    # Unset the `cache_dir` option
    config_str = _regex_replace("cache_dir = .*", "cache_dir = ", config_str)

    # Set the JIT compiler to gcc
    config_str = _regex_replace("compiler_cmd = \".* -x c", "compiler_cmd = \"gcc -x c", config_str)

    # Compile command: replace absolute include path with a path relative to {CONF_PATH}.
    config_str = _regex_replace("-I%s/share/bohrium/" % args_extra.bh_install_prefix, "-I{CONF_PATH}/", config_str)

    # Compile command: replace absolute library path with a path relative to {CONF_PATH}.
    config_str = _regex_replace("-L%s/" % args_extra.bh_install_prefix, "-L{CONF_PATH}/", config_str)

    # Replace absolute library paths with a relative path.
    config_str = _regex_replace("%s/" % args_extra.bh_install_prefix, "./", config_str)
    f.write(config_str)
    print("writing config file: %s" % f.name)


# Let's find the version
cmd = "git describe --tags --long --match v[0-9]*"
print(cmd)
try:
    # Let's get the Bohrium version without the 'v' and hash (e.g. v0.8.9-47-g6464 => v0.8.9-47)
    _version = subprocess.check_output(cmd, shell=True, cwd=join(_script_path(), '..', '..'))
    print("_version: '%s'" % str(_version))
    _version = re.match(".*v(.+-\d+)-", str(_version)).group(1)
except subprocess.CalledProcessError as e:
    print("Couldn't find the Bohrium version through `git describe`, are we not in the git repos?\n"
          "Using the VERSION file instead")
    # The version if written in the VERSION file in the root of Bohrium
    with open(join(_script_path(), "..", "..", "VERSION"), "r") as f:
        _version = f.read().strip()
print("Bohrium version: %s" % _version)


# Get the long description from the README file
with open(os.path.join(_script_path(), '../../README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bohrium',
    version=_version,
    description='Bohrium NumPy',
    long_description=long_description,
    author='The Bohrium Team',
    author_email='contact@bh107.org',
    url='http://www.bh107.org',
    license='LGPLv3',
    maintainer='Mads R. B. Kristensen',
    maintainer_email='madsbk@gmail.com',
    platforms=['Linux', 'OSX'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',

        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: Python :: Implementation :: CPython'
    ],

    # What does your project relate to?
    keywords='Bohrium, bh107, Python, C, CUDA, OpenCL',

    package_dir={'bohrium': args_extra.npbackend_dir},
    packages=['bohrium', 'bohrium.target', 'bohrium.nobh'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.13'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
   # extras_require={
   #     'dev': ['check-manifest'],
   #     'test': ['coverage'],
   # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'bohrium': _find_data_files(args_extra.npbackend_dir, regex_exclude="\.pyc|\.py") +
                   _find_data_files(args_extra.npbackend_dir,
                                    regex_include="test/run\.py|test/util\.py|test/tests/test.+\.py")
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    #entry_points={
    #    'console_scripts': [
    #        'bp-run=benchpress.run:main',
    #    ],
    #},
)

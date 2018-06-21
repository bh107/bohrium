#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Creates a PIP wheel of Bohrium"""

from setuptools import setup
from distutils.dir_util import mkpath, copy_tree
from wheel.bdist_wheel import bdist_wheel, pep425tags
from codecs import open  # To use a consistent encoding
import os
import re
import argparse
import sys
import shutil
import glob
import subprocess
import platform
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
    '--lib-dir-name',
    type=str,
    default='lib64',
    help='Name of the shared library folder'
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
            if ".dylib" in fname:
                # We need this HACK because osx might not preserve the write and exec permission
                out_path = join(dst_dir, os.path.basename(fname))
                shutil.copyfile(fname, out_path)
                subprocess.check_call("chmod a+x %s" % out_path, shell=True)
            else:
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


def _update_rpath(filename_list, rpath):
    """Update the RPATH of the files in `filename_list`"""

    if platform.system() == "Darwin":
        return

    for filename in filename_list:
        cmd = "patchelf --set-rpath '%s' %s" % (rpath, filename)
        print(cmd)
        subprocess.check_call(cmd, shell=True)


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
_copy_files(join(args_extra.bh_install_prefix, args_extra.lib_dir_name, 'lib*'),
            join(args_extra.npbackend_dir, "lib64"))


# Copy extra libraries specified by the user
if args_extra.lib is not None:
    for lib in args_extra.lib:
        _copy_files(lib, join(args_extra.npbackend_dir, "lib64"))


# Copy extra libraries specified by the user
if args_extra.bin is not None:
    for bin in args_extra.bin:
        _copy_files(bin, join(args_extra.npbackend_dir, "bin"))


py_sos = glob.glob(join(args_extra.npbackend_dir, '*.so'))
lib64_files = glob.glob(join(args_extra.npbackend_dir, 'lib64', '*'))
bin_files = glob.glob(join(args_extra.npbackend_dir, 'bin', '*'))
if platform.system() == "Linux":
    # Update the RPATH of the Python extensions to look in the the `lib64` dir
    _update_rpath(py_sos, '$ORIGIN/lib64')
    # Update the RPATH of Bohrium's shared libraries to look in the current dir
    _update_rpath(lib64_files, '$ORIGIN')
    # Update the RPATH of Bohrium's binaries to look in the current dir
    _update_rpath(bin_files, '$ORIGIN/../lib64')
elif platform.system() == "Darwin":
    all_files = {}
    for file_path in py_sos + lib64_files + bin_files:
        file_name = os.path.basename(file_path)
        assert(file_name not in all_files)
        all_files[file_name] = file_path

    for file_path in py_sos + lib64_files + bin_files:
        cmd = "otool -L %s" % (file_path)
        otool_res = subprocess.check_output(cmd, shell=True).decode('utf-8')
        for line in otool_res.splitlines()[1:]:  # Each line in `otool_res` represents a linking path (except 1. line)
            dylib_path = line.strip().split()[0]
            dylib_name = os.path.basename(dylib_path)
            if os.path.basename(dylib_path) in all_files:
                load_path = " @loader_path%s" % all_files[dylib_name].replace(os.path.dirname(file_path), "")
                cmd = "install_name_tool -change %s %s %s" % (dylib_path, load_path, file_path)
                subprocess.check_output(cmd, shell=True)

# Write a modified config file to the Python package dir
_config_path = join(args_extra.npbackend_dir, "config.ini")
_config_dir = os.path.dirname(_config_path)
with open(_config_path, "w") as f:
    config_str = args_extra.config.read()

    # Unset the `cache_dir` option
    config_str = _regex_replace("cache_dir = .*", "cache_dir = ~/.bohrium/cache", config_str)

    # Set the JIT compiler to gcc
    config_str = _regex_replace("compiler_cmd = \".* -x c", "compiler_cmd = \"gcc -x c", config_str)

    # clang doesn't support some unneeded flags
    config_str = _regex_replace("-Wno-expansion-to-defined", "", config_str)
    config_str = _regex_replace("-Wno-pass-failed", "", config_str)

    # Replace `lib` with `lib64` since we always includes shared libraries in `lib64`
    config_str = _regex_replace("%s/lib/" % args_extra.bh_install_prefix,
                                "%s/lib64/" % args_extra.bh_install_prefix, config_str)
    config_str = _regex_replace("%s/lib " % args_extra.bh_install_prefix,
                                "%s/lib64 " % args_extra.bh_install_prefix, config_str)

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
    _version = subprocess.check_output(cmd, shell=True, cwd=join(_script_path(), '..', '..')).decode('utf-8')
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
with open(os.path.join(_script_path(), '../../README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# We have to manually set the package tags. At this point, setuptools thinks that this is a pure python package.
class taged_bdist_wheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        impl_name = pep425tags.get_abbr_impl()
        impl_ver = pep425tags.get_impl_ver()
        abi_tag = str(pep425tags.get_abi_tag()).lower()
        if platform.system() == "Linux":  # Building on Linux, we assume `manylinux1_x86_64`
            plat_name = "manylinux1_x86_64"
        else:
            plat_name = pep425tags.get_platform()
        return (impl_name + impl_ver, abi_tag, plat_name)


# For now, only the osx packages requires 'gcc7'
install_requires = ['numpy>=1.13']
if platform.system() == "Darwin":
    install_requires.append('gcc7')


# Finally, we call the setup
setup(
    cmdclass={'bdist_wheel': taged_bdist_wheel},
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
    packages=['bohrium', 'bohrium.nobh'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,

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

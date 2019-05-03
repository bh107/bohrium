#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numbers
import re
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as setup_sdist

def script_path(*paths):
    prefix = os.path.abspath(os.path.dirname(__file__))
    assert len(prefix) > 0
    return os.path.join(prefix, *paths)

def version_file_exist():
    """Return whether the version.py file exist or not"""
    ver_path = script_path("bh107", "version.py")
    return os.path.exists(ver_path)


def get_version():
    """Returns the version and version_info.
        If the version.py file doesn't exist, the version of Bohrium API is returned.
        NB: If the version.py file doesn't exist, this function must be called after the call to `setup()`.
    """
    ver_path = script_path("bh107", "version.py")
    if os.path.exists(ver_path):
        print("Getting version from version.py")
        # Loading `__version__` variable from the version file
        with open(ver_path, "r") as f:
            t=f.read()
            version = re.search("__version__\s*=\s*\"([^\"]*)\"",t).group(1);
            version_info = eval(re.search("__version_info__\s*=\s*(\([^\)]+\))",t).group(1));
            return (version, version_info)
    else:
        print("Getting version from bohrium_api")
        import bohrium_api
        return (bohrium_api.__version__, bohrium_api.__version_info__)


def get_bohrium_api_required_string():
    """Returns the install_requires/setup_requires string for `bohrium_api`"""
    try:
        ver_tuple = get_version()[1]
        return "bohrium_api>=%d.%d.%d" % (ver_tuple[0], ver_tuple[1], ver_tuple[2])
    except ImportError:
        return "bohrium_api"  # If `bohrium_api` is not available, we expect PIP to install the newest package


def gen_version_file_in_cmd(self, target_dir):
    """We extend the setup commands to also generate the `version.py` file if it doesn't exist already"""
    if not self.dry_run:
        version, version_info = get_version()
        if not version_file_exist():
            self.mkpath(target_dir)
            p = os.path.join(target_dir, 'version.py')
            print("Generating '%s'" % p)
            with open(p, 'w') as fobj:
                fobj.write("__version__ = \"%s\"\n" % version)
                fobj.write("__version_info__ = %s\n" % str(version_info))


class Sdist(setup_sdist):
    def run(self):
        gen_version_file_in_cmd(self, "bh107")
        setup_sdist.run(self)

class DelayedVersion(numbers.Number):
    """In order to delay the version evaluation that depend on `bohrium_api`, we use this class"""

    def __str__(self):
        return get_version()[0]


setup(
    name='bh107',

    cmdclass={'sdist': Sdist},
    version=DelayedVersion(),

    description='Bohrium for Python <www.bh107.org>',
    long_description='Bohrium for Python <www.bh107.org>',

    # The project's main homepage.
    url='http://bh107.org',

    # Author details
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
        'Intended Audience :: End Users/Desktop',
        'Topic :: Multimedia',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='Bohrium, bh107, pyopencl, pycuda',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'bohrium_api'],
)

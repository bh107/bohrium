#!/usr/bin/env python
import subprocess
from subprocess import check_output, check_call, Popen, PIPE, STDOUT
from datetime import datetime
import os
from os import path
import argparse
import tempfile
import re
import traceback
import shutil


CONTROL ="""\
Source: bohrium
Section: devel
Priority: optional
Maintainer: Bohrium Builder <builder@bh107.org>
Build-Depends: python-dev, python-numpy, cython, python3-dev, python3-numpy, python3-dev, cython3, debhelper, cmake, swig, fftw3-dev, ocl-icd-opencl-dev, libgl-dev, libboost-serialization-dev, libboost-filesystem-dev, libboost-system-dev, libboost-regex-dev, libhwloc-dev, freeglut3-dev, libxmu-dev, libxi-dev, zlib1g-dev, libopenblas-dev, liblapack-dev, liblapacke-dev, libclblas-dev
Standards-Version: 3.9.5
Homepage: http://www.bh107.org

Package: bohrium
Architecture: amd64
Depends: build-essential, libboost-dev, python (>= 2.7), python-numpy (>= 1.8), fftw3, libboost-serialization-dev, libboost-filesystem-dev, libboost-system-dev, libboost-regex-dev, libhwloc-dev, libopenblas-dev, liblapack-dev, liblapacke-dev
Recommends:
Suggests: bohrium-opencl, bohrium-visualizer, ipython,
Description:  Bohrium Runtime System: Automatic Array Parallelization in C, C++, and Python

Package: bohrium3
Architecture: amd64
Depends: bohrium, python3 (>= 3.4), python3-numpy (>= 1.8)
Recommends:
Suggests: bohrium-opencl, bohrium-visualizer, ipython,
Description:  The Python v3 frontend for the Bohrium Runtime System

Package: bohrium-opencl
Architecture: amd64
Depends: bohrium, opencl-dev, libopencl1, libgl-dev, libclblas-dev
Recommends:
Suggests: bumblebee
Description: The GPU (OpenCL) backend for the Bohrium Runtime System

Package: bohrium-visualizer
Architecture: amd64
Depends: bohrium, freeglut3, libxmu6, libxi6
Recommends:
Suggests:
Description: The Visualizer for the Bohrium Runtime System

"""

# Ubuntu Trusty doesn't have the `libclblas-dev` package
CONTROL_TRUSTY = CONTROL.replace(", libclblas-dev", "")

RULES ="""\
#!/usr/bin/make -f

build:
	mkdir b
	cd b; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
	$(MAKE) VERBOSE=1 -C b preinstall
	touch build

binary-core: build
	cd b; cmake -DCOMPONENT=bohrium -DCMAKE_INSTALL_PREFIX=../debian/core/usr -P cmake_install.cmake
	mv debian/core/usr/lib/python2.7/site-packages debian/core/usr/lib/python2.7/dist-packages
	mkdir -p debian/core/DEBIAN
	cp -p debian/preinst debian/core/DEBIAN/
	cp -p debian/prerm debian/core/DEBIAN/
	dpkg-gensymbols -q -pbohrium -Pdebian/core
	dpkg-gencontrol -pbohrium -Pdebian/core
	dpkg --build debian/core ..

binary-core3: build
	cd b; cmake -DCOMPONENT=bohrium3 -DCMAKE_INSTALL_PREFIX=../debian/core3/usr -P cmake_install.cmake
	cd debian/core3/usr/lib/python3*/; mv ./site-packages/ ./dist-packages/
	mkdir -p debian/core3/DEBIAN
	dpkg-gensymbols -q -pbohrium3 -Pdebian/core3
	dpkg-gencontrol -pbohrium3 -Pdebian/core3
	dpkg --build debian/core3 ..

binary-opencl: build
	cd b; cmake -DCOMPONENT=bohrium-opencl -DCMAKE_INSTALL_PREFIX=../debian/opencl/usr -P cmake_install.cmake
	mkdir -p debian/opencl/DEBIAN
	dpkg-gensymbols -q -pbohrium-opencl -Pdebian/opencl
	dpkg-gencontrol -pbohrium-opencl -Pdebian/opencl -Tdebian/bohrium-opencl.substvars
	dpkg --build debian/opencl ..

binary-visualizer: build
	cd b; cmake -DCOMPONENT=bohrium-visualizer -DCMAKE_INSTALL_PREFIX=../debian/visualizer/usr -P cmake_install.cmake
	mkdir -p debian/visualizer/DEBIAN
	dpkg-gensymbols -q -pbohrium-visualizer -Pdebian/visualizer
	dpkg-gencontrol -pbohrium-visualizer -Pdebian/visualizer -Tdebian/bohrium-visualizer.substvars
	dpkg --build debian/visualizer ..

binary: binary-indep binary-arch

binary-indep: build

binary-arch: binary-core binary-core3 binary-opencl binary-visualizer

clean:
	rm -f build
	rm -rf b

.PHONY: binary binary-arch binary-indep clean
"""

REMOVE_CACHEFILES = """\
#!/bin/sh

set -e

rm -fR /usr/var/bohrium/fuse_cache/*
rm -fR /usr/var/bohrium/kernels/*
rm -fR /usr/var/bohrium/objects/*
rm -fR /usr/var/bohrium/source/*
rm -fR /usr/var/bohrium/object/*
exit 0
"""


UBUNTU_RELEASES = ['trusty', 'xenial', 'zesty']


SRC = path.join(path.dirname(os.path.realpath(__file__)),"..","..")

def bash_cmd(cmd, cwd=None):
    print cmd
    out = ""
    try:
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True, cwd=cwd)
        while p.poll() is None:
            t = p.stdout.readline()
            out += t
            print t,
        t = p.stdout.read()
        out += t
        print t,
        p.wait()
    except KeyboardInterrupt:
        p.kill()
        raise
    return out

def build_src_dir(args, bh_version, release="trusty"):
    global SRC
    deb_src_dir = "%s/bohrium-%s-ubuntu1~%s1/debian"%(args.output, bh_version, release)
    os.makedirs(deb_src_dir)

    bash_cmd("tar -xzf  %s/bohrium_%s.orig.tar.gz"%(args.output, bh_version), cwd="%s/.."%deb_src_dir)

    #debian/control
    with open("%s/control"%deb_src_dir, "w") as f:
        if release == "trusty": # TODO: remove when we are not building for Trusty anymore.
            f.write(CONTROL_TRUSTY)
        else:
            f.write(CONTROL)

    #debian/rules
    with open("%s/rules"%deb_src_dir, "w") as f:
        f.write(RULES)
    bash_cmd("chmod +x %s/rules"%deb_src_dir)

    #debian/compat
    with open("%s/compat"%deb_src_dir, "w") as f:
        f.write("7")

    #debian/preinst
    with open("%s/preinst"%deb_src_dir, "w") as f:
        f.write(REMOVE_CACHEFILES)
    bash_cmd("chmod +x %s/preinst"%deb_src_dir)

    #debian/prerm
    with open("%s/prerm"%deb_src_dir, "w") as f:
        f.write(REMOVE_CACHEFILES)
    bash_cmd("chmod +x %s/prerm"%deb_src_dir)

    #debian/source/format
    os.makedirs(path.join(deb_src_dir,"source"))
    with open("%s/source/format"%deb_src_dir, "w") as f:
        f.write("3.0 (quilt)")

    #debian/copyright
    shutil.copy(path.join(SRC,"COPYING"), path.join(deb_src_dir,"copyright"))

    #debian/changelog
    date = bash_cmd("date -R")
    with open("%s/changelog"%deb_src_dir, "w") as f:
        t  = "bohrium (%s-ubuntu1~%s1) %s; urgency=medium\n\n"%(bh_version, release, release)
        t += "  * Nightly package build\n\n"
        t += " -- Bohrium Builder <builder@bh107.org>  %s"%(date)
        f.write(t)

    #Check if we should sign the package
    if args.unsign:
        unsign = "-us -uc"
    else:
        unsign = ""

    #Let's build the package
    bash_cmd("debuild -S %s"%unsign, cwd=deb_src_dir)


def main(args):
    global SRC

    #Lets get the Bohrium version without the 'v' char (e.g. v0.2-0-g6a2352d => 0.2-0-g6a2352d)
    bh_version = bash_cmd("git describe --tags --long --match *v[0-9]* ", cwd=SRC)
    bh_version = bh_version.strip()[1:]

    #Get source archive
    bash_cmd("mkdir -p %s"%args.output)
    bash_cmd("git archive --format=tar.gz -o %s/bohrium_%s.orig.tar.gz HEAD"%(args.output, bh_version), cwd=SRC)

    #Lets build a source dir for each Ubuntu Release
    for release_name in UBUNTU_RELEASES:
        build_src_dir(args, bh_version, release_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Build the debian source packages.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--contact',
        default="Bohrium Builder <builder@bh107.org>",
        type=str,
        help='The package contact info which is also used for signing the package.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='The output directory.'
    )
    parser.add_argument(
        '--unsign',
        action='store_true',
        help="Use if the package shouldn't be signed."
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = tempfile.mkdtemp()
    args.output = os.path.abspath(args.output)

    print "output dir: ", args.output
    main(args)

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
Build-Depends: python-numpy, debhelper, cmake, swig, python-cheetah, python-dev, fftw3-dev, cython, ocl-icd-opencl-dev, libgl-dev, libboost-serialization-dev, libboost-system-dev, libboost-filesystem-dev, libboost-thread-dev, mono-devel, mono-gmcs, libhwloc-dev, freeglut3-dev, libxmu-dev, libxi-dev
Standards-Version: 3.9.5
Homepage: http://www.bh107.org

Package: bohrium
Architecture: amd64
Depends: build-essential, libboost-dev, python (>= 2.7), python-numpy (>= 1.6), fftw3, libboost-serialization-dev, libboost-system-dev, libboost-filesystem-dev, libboost-thread-dev, libhwloc-dev
Recommends:
Suggests: bohrium-gpu, ipython,
Description:  Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python

Package: bohrium-numcil
Architecture: amd64
Depends: bohrium, mono-devel, mono-gmcs
Recommends:
Suggests:
Description: The NumCIL (.NET) frontend for the Bohrium Runtime System

Package: bohrium-gpu
Architecture: amd64
Depends: bohrium, opencl-dev, libopencl1, libgl-dev
Recommends:
Suggests: bumblebee
Description: The GPU (OpenCL) backend for the Bohrium Runtime System

"""

RULES ="""\
#!/usr/bin/make -f

build:
	mkdir b
	cd b; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DVEM_CLUSTER=OFF ..
	$(MAKE) VERBOSE=1 -C b preinstall
	touch build

binary-core: build
	cd b; cmake -DCOMPONENT=bohrium -DCMAKE_INSTALL_PREFIX=../debian/core/usr -P cmake_install.cmake
	mv debian/core/usr/lib/python2.7/site-packages debian/core/usr/lib/python2.7/dist-packages
	mkdir -p debian/core/DEBIAN
	dpkg-gensymbols -q -pbohrium -Pdebian/core
	dpkg-gencontrol -pbohrium -Pdebian/core
	dpkg --build debian/core ..
        cp -p preinst debian/core/DEBIAN/
        cp -p prerm debian/core/DEBIAN/

binary-numcil: build
	cd b; cmake -DCOMPONENT=bohrium-numcil -DCMAKE_INSTALL_PREFIX=../debian/numcil/usr -P cmake_install.cmake
	mkdir -p debian/numcil/DEBIAN
	dpkg-gensymbols -q -pbohrium-numcil -Pdebian/numcil
	dpkg-gencontrol -pbohrium-numcil -Pdebian/numcil -Tdebian/bohrium-numcil.substvars
	dpkg --build debian/numcil ..

binary-gpu: build
	cd b; cmake -DCOMPONENT=bohrium-gpu -DCMAKE_INSTALL_PREFIX=../debian/gpu/usr -P cmake_install.cmake
	mkdir -p debian/gpu/DEBIAN
	dpkg-gensymbols -q -pbohrium-gpu -Pdebian/gpu
	dpkg-gencontrol -pbohrium-gpu -Pdebian/gpu -Tdebian/bohrium-gpu.substvars
	dpkg --build debian/gpu ..

binary: binary-indep binary-arch

binary-indep: build

binary-arch: binary-core binary-gpu binary-numcil

clean:
	rm -f build
	rm -rf b

.PHONY: binary binary-arch binary-indep clean
"""

REMOVE_CACHEFILES = """\
#!/bin/sh

set -e
echo "Cleanup old Bohrium cache files"
rm -f /usr/var/bohrium/fuse_cache/*
rm -f /usr/var/bohrium/kernels/*
rm -f /usr/var/bohrium/objects/*

exit 0
"""

UBUNTU_RELEASES = ['trusty', 'vivid']


SRC = path.join(path.dirname(os.path.realpath(__file__)),"..","..")

def bash_cmd(cmd, cwd=None):
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

    bash_cmd("debuild -S", cwd=deb_src_dir)


def main(args):
    global SRC

    #Lets get the Bohrium version without the 'v' char (e.g. v0.2-0-g6a2352d => 0.2-0-g6a2352d)
    bh_version = bash_cmd("git describe --tags --long --match *v[0-9]* ", cwd=SRC)
    bh_version = bh_version.strip()[1:]

    #Get source archive
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
    args = parser.parse_args()
    if args.output is None:
        args.output = tempfile.mkdtemp()

    print "output dir: ", args.output
    main(args)


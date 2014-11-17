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
Build-Depends: python-numpy, debhelper, cmake, swig, libctemplate-dev, libboost-dev, python-cheetah, python-dev, fftw3-dev, cython, ocl-icd-opencl-dev, libgl-dev, mpich2, libmpich2-dev, libopenmpi-dev, openmpi-bin
Standards-Version: 3.9.5
Homepage: http://www.bh107.org

Package: bohrium
Architecture: amd64
Depends: libctemplate-dev, build-essential, libboost-dev, python (>= 2.7), python-numpy (>= 1.6), fftw3,
Recommends:
Suggests: bohrium-gpu, bohrium-mpich, bohrium-openmpi, ipython,
Description:  Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python

Package: bohrium-gpu
Architecture: amd64
Depends: bohrium, opencl-dev, libopencl1, libgl-dev
Recommends:
Suggests: bumblebee
Description: The GPU (OpenCL) backend for the Bohrium Runtime System

Package: bohrium-mpich
Architecture: amd64
Depends: bohrium, mpich2
Conflicts: bohrium-openmpi
Recommends:
Suggests:
Description: The Cluster (MPICH) backend for the Bohrium Runtime System

Package: bohrium-openmpi
Architecture: amd64
Depends: bohrium, openmpi-bin
Conflicts: bohrium-mpich
Recommends:
Suggests:
Description: The Cluster (OpenMPI) backend for the Bohrium Runtime System
"""

RULES ="""\
#!/usr/bin/make -f

build:
	mkdir b
	cd b; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DNO_VEM_CLUSTER=1 ..
	$(MAKE) VERBOSE=1 -C b preinstall
	touch build

binary-core: build
	cd b; cmake -DCOMPONENT=bohrium -DCMAKE_INSTALL_PREFIX=../debian/core/usr -P cmake_install.cmake
	mv debian/core/usr/lib/python2.7/site-packages debian/core/usr/lib/python2.7/dist-packages
	mkdir -p debian/core/DEBIAN
	dpkg-gensymbols -q -pbohrium -Pdebian/core
	dpkg-gencontrol -pbohrium -Pdebian/core
	dpkg --build debian/core ..

binary-gpu: build
	cd b; cmake -DCOMPONENT=bohrium-gpu -DCMAKE_INSTALL_PREFIX=../debian/gpu/usr -P cmake_install.cmake
	mkdir -p debian/gpu/DEBIAN
	dpkg-gensymbols -q -pbohrium-gpu -Pdebian/gpu
	dpkg-gencontrol -pbohrium-gpu -Pdebian/gpu -Tdebian/bohrium-gpu.substvars
	dpkg --build debian/gpu ..

binary-mpich: build
	rm -Rf b/vem/cluster
	cd b; cmake -UMPI* -DNO_VEM_CLUSTER=1 ..
	cd b; cmake -UNO_VEM_CLUSTER -DMPI_CXX_COMPILER=mpicxx.mpich2 -DMPI_C_COMPILER=mpicc.mpich2 ..
	$(MAKE) VERBOSE=1 -C b preinstall
	cd b; cmake -DCOMPONENT=bohrium-cluster -DCMAKE_INSTALL_PREFIX=../debian/mpich/usr -P cmake_install.cmake
	mkdir -p debian/mpich/DEBIAN
	dpkg-gensymbols -q -pbohrium-mpich -Pdebian/mpich
	dpkg-gencontrol -pbohrium-mpich -Pdebian/mpich -Tdebian/bohrium-mpich.substvars
	dpkg --build debian/mpich ..

binary-openmpi: build
	rm -Rf b/vem/cluster
	cd b; cmake -UMPI* -DNO_VEM_CLUSTER=1 ..
	cd b; cmake -UNO_VEM_CLUSTER -DMPI_CXX_COMPILER=mpicxx.openmpi -DMPI_C_COMPILER=mpicc.openmpi ..
	$(MAKE) VERBOSE=1 -C b preinstall
	cd b; cmake -DCOMPONENT=bohrium-cluster -DCMAKE_INSTALL_PREFIX=../debian/openmpi/usr -P cmake_install.cmake
	mkdir -p debian/openmpi/DEBIAN
	dpkg-gensymbols -q -pbohrium-openmpi -Pdebian/openmpi
	dpkg-gencontrol -pbohrium-openmpi -Pdebian/openmpi -Tdebian/bohrium-openmpi.substvars
	dpkg --build debian/openmpi ..

binary: binary-indep binary-arch

binary-indep: build

binary-arch: binary-core binary-gpu binary-mpich binary-openmpi

clean:
	rm -f build
	rm -rf b

.PHONY: binary binary-arch binary-indep clean
"""

UBUNTU_RELEASES = ['trusty', 'utopic']


SRC = path.join(path.dirname(os.path.realpath(__file__)),"..","..")

def bash_cmd(cmd, cwd=None):
    print cmd
    p = subprocess.Popen(
        cmd,
        stdout  = subprocess.PIPE,
        stderr  = subprocess.PIPE,
        shell = True,
        cwd=cwd
    )
    out, err = p.communicate()
    print out,
    print err,
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


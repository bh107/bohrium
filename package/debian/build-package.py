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

def main(args):
    global SRC

    version = bash_cmd("git describe --tags --long --match *v[0-9]* ", cwd=SRC)
    #Lets remove the 'v' char in the version tag (e.g. v0.2-0-g6a2352d => 0.2-0-g6a2352d)
    version = version.strip()[1:]

    bash_cmd("git archive --format=tar.gz -o %s/bohrium_%s.orig.tar.gz HEAD"%(args.output, version), cwd=SRC)

    deb_src_dir = "%s/bohrium-%s-ubuntu1~trusty1/debian"%(args.output, version)
    os.makedirs(deb_src_dir)

    bash_cmd("tar -xzf  %s/bohrium_%s.orig.tar.gz"%(args.output, version), cwd="%s/.."%deb_src_dir)

    #debian/control
    with open("%s/control"%deb_src_dir, "w") as f:
        t = """\
Source: bohrium
Section: devel
Priority: optional
Maintainer: Bohrium Builder <builder@bh107.org>
Build-Depends: python-numpy, debhelper, cmake, swig, libctemplate-dev, libboost-dev, python-cheetah, python-dev, fftw3-dev, cython, ocl-icd-opencl-dev, libgl-dev, mpich2, libmpich2-dev
Standards-Version: 3.9.5
Homepage: http://www.bh107.org

Package: bohrium
Architecture: amd64
Depends: libctemplate-dev, build-essential, libboost-dev, python (>= 2.7), python-numpy (>= 1.6), fftw3,
Recommends:
Suggests: bohrium-gpu, bohrium-mpich, ipython,
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
Recommends:
Suggests:
Description: The Cluster (MPICH) backend for the Bohrium Runtime System
"""
        f.write(t)

    #debian/rules
    with open("%s/rules"%deb_src_dir, "w") as f:
        t = """\
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
	dpkg-gensymbols -q -pbohrium -Pdebian/core
#	dh_shlibdeps
#	dh_strip
	dpkg-gencontrol -pbohrium -Pdebian/core
	dpkg --build debian/core ..

binary-gpu: build
	cd b; cmake -DCOMPONENT=bohrium-gpu -DCMAKE_INSTALL_PREFIX=../debian/gpu/usr -P cmake_install.cmake
	mkdir -p debian/gpu/DEBIAN
	dpkg-gensymbols -q -pbohrium-gpu -Pdebian/gpu
#	dh_shlibdeps
#	dh_strip
	dpkg-gencontrol -pbohrium-gpu -Pdebian/gpu -Tdebian/bohrium-gpu.substvars
	dpkg --build debian/gpu ..

binary-mpich: build
	cd b; cmake -DCOMPONENT=bohrium-cluster -DCMAKE_INSTALL_PREFIX=../debian/mpich/usr -P cmake_install.cmake
	mkdir -p debian/mpich/DEBIAN
	dpkg-gensymbols -q -pbohrium-mpich -Pdebian/mpich
#	dh_shlibdeps
#	dh_strip
	dpkg-gencontrol -pbohrium-mpich -Pdebian/mpich -Tdebian/bohrium-mpich.substvars
	dpkg --build debian/mpich ..

binary: binary-indep binary-arch

binary-indep: build

binary-arch: binary-core binary-mpich binary-gpu

clean:
	rm -f build
	rm -rf b

.PHONY: binary binary-arch binary-indep clean
"""
        f.write(t)
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
        t  = "bohrium (%s-ubuntu1~trusty1) trusty; urgency=medium\n\n"%(version)
        t += "  * Nightly package build\n\n"
        t += " -- Bohrium Builder <builder@bh107.org>  %s"%(date)
        f.write(t)

    bash_cmd("debuild -S", cwd=deb_src_dir)




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

    status = "SUCCESS"
    try:
        print "output dir: ", args.output
        main(args)
    except StandardError, e:
        out += "*"*70
        out += "\nERROR: %s"%traceback.format_exc()
        out += "*"*70
        out += "\n"
        status = "FAILURE"
        try:
            out += e.output
        except:
            pass


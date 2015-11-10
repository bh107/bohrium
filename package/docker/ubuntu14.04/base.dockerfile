# This is a Dockerfile for installing Bohrium dependencies

FROM ubuntu:14.04
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>
RUN mkdir -p /bohrium/build
WORKDIR /bohrium/build

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -qq wget unzip build-essential
RUN apt-get install -qq cmake swig python python-numpy python-cheetah python-dev cython
RUN apt-get install -qq libboost-serialization-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev
RUN apt-get install -qq mono-mcs mono-xbuild libmono-system-numerics4.0-cil libmono-microsoft-build-tasks-v4.0-4.0-cil
RUN apt-get install -qq fftw3-dev
RUN apt-get install -qq libhwloc-dev
RUN apt-get install -qq libgl1-mesa-dev

# Install AMD SDK for OpenCL
RUN mkdir -p /opt/amd_src
WORKDIR /opt/amd_src
ENV OPENCL_HOME /opt/AMDAPPSDK-2.9-1
ENV OPENCL_LIBPATH /opt/AMDAPPSDK-2.9-1/lib/x86_64
RUN wget -nv http://jenkins.choderalab.org/userContent/AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2; exit 0;
RUN tar xjf AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2
RUN ./AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh -- -s -a yes
ENV OpenCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
ENV OpenCL_INCPATH "/opt/AMDAPPSDK-2.9-1/include"
ENV LD_LIBRARY_PATH "$OpenCL_LIBPATH:$LD_LIBRARY_PATH"

# Install debug dependencies
RUN apt-get install -qq zlib1g-dev valgrind gdb vim cgdb

# Build and install source dependencies
RUN mkdir -p /opt/dython
WORKDIR /opt/dython
ENV PV 2.7.3
RUN wget -q http://www.python.org/ftp/python/$PV/Python-$PV.tgz
RUN tar -xzf Python-$PV.tgz
WORKDIR Python-$PV
RUN ./configure --with-pydebug --without-pymalloc --with-valgrind --prefix /opt/python
RUN make install
RUN ln -s /opt/python/bin/python /usr/bin/dython

RUN mkdir -p /opt/cython
WORKDIR /opt/cython
ENV CV 0.22
RUN wget -q http://cython.org/release/Cython-$CV.tar.gz
RUN tar -xzf Cython-$CV.tar.gz
WORKDIR Cython-$CV
RUN dython setup.py install

RUN mkdir -p /opt/cheetah
WORKDIR /opt/cheetah
ENV CTV 2.4.4
RUN wget -q https://pypi.python.org/packages/source/C/Cheetah/Cheetah-$CTV.tar.gz
RUN tar -xzf Cheetah-$CTV.tar.gz
WORKDIR Cheetah-$CTV
RUN dython setup.py install

RUN mkdir -p /opt/numpy
WORKDIR /opt/numpy
ENV NV 1.8.2
RUN wget -q http://optimate.dl.sourceforge.net/project/numpy/NumPy/$NV/numpy-$NV.tar.gz
RUN tar -xzf numpy-$NV.tar.gz
WORKDIR numpy-$NV
RUN dython setup.py install

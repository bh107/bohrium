# This is a Dockerfile for installing Bohrium dependencies

FROM ubuntu:16.04
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
RUN apt-get install -qq libboost-serialization-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-regex-dev
RUN apt-get install -qq mono-mcs mono-xbuild libmono-system-numerics4.0-cil libmono-microsoft-build-tasks-v4.0-4.0-cil
RUN apt-get install -qq libblas-dev liblapack-dev libclblas-dev
RUN apt-get install -qq fftw3-dev
RUN apt-get install -qq libhwloc-dev
RUN apt-get install -qq libgl1-mesa-dev
RUN apt-get install -qq python3 python3-numpy python3-dev cython3
RUN apt-get install -qq python2.7-scipy python3-scipy

# Install AMD SDK for OpenCL
RUN mkdir -p /opt/amd_src
WORKDIR /opt/amd_src
ENV OPENCL_HOME /opt/AMDAPPSDK-2.9-1
ENV OPENCL_LIBPATH /opt/AMDAPPSDK-2.9-1/lib/x86_64

# RUN wget -nv http://jenkins.choderalab.org/userContent/AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2; exit 0;
COPY AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2 .

RUN tar xjf AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2
RUN ./AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh -- -s -a yes
ENV OpenCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
ENV OpenCL_INCPATH "/opt/AMDAPPSDK-2.9-1/include"
ENV LD_LIBRARY_PATH "$OpenCL_LIBPATH:$LD_LIBRARY_PATH"

# Install debug dependencies
RUN apt-get install -qq zlib1g-dev valgrind gdb vim cgdb

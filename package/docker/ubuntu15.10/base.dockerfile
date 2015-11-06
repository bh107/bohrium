# This is a Dockerfile for installing Bohrium dependencies

FROM ubuntu:15.10
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
RUN apt-get install -qq ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers
#RUN apt-get install -qq fglrx fglrx-dev
#RUN apt-get install -qq freeglut3 freeglut3-dev libxmu-dev libxi-dev

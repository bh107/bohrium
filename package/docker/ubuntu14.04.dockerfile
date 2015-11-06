# This is a Dockerfile for installing and testing Bohrium
# Please make sure that the build "context" is pointing to the root of Bohrium source files
# e.g. 'docker build -t bohrium -f <path to this file> <path to bohrium source>'
# Then you can run 'docker run -t bohrium' to Bohrium test

FROM ubuntu:14.04
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>
RUN mkdir -p /tmp/bohrium/build
WORKDIR /tmp/bohrium/build

# Install dependencies
RUN apt-get update
RUN apt-get install -qq wget unzip build-essential
RUN apt-get install -qq cmake swig python python-numpy python-cheetah python-dev cython
RUN apt-get install -qq libboost-serialization-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev
RUN apt-get install -qq mono-mcs mono-xbuild libmono-system-numerics4.0-cil libmono-microsoft-build-tasks-v4.0-4.0-cil
RUN apt-get install -qq fftw3-dev
RUN apt-get install -qq libhwloc-dev
RUN apt-get install -qq ocl-icd-opencl-dev ocl-icd-libopencl1
RUN apt-get install -qq fglrx fglrx-dev opencl-headers
RUN apt-get install -qq freeglut3 freeglut3-dev libxmu-dev libxi-dev

# Copy and build bohrium source files from "context"
COPY . ../
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN make
RUN make install

# Test Suite
ENV PYTHONPATH /usr/lib/python2.7/site-packages
ENTRYPOINT echo $BH_STACK && python /tmp/bohrium/test/python/numpytest.py

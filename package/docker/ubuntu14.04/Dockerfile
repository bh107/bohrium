FROM ubuntu:14.04
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>
RUN mkdir -p /tmp/bohrium/build
WORKDIR /tmp/bohrium/build

# Install dependencies
RUN apt-get update
RUN apt-get install -y wget unzip build-essential
RUN apt-get install -y cmake swig python python-numpy python-cheetah python-dev cython
RUN apt-get install -y libboost-serialization-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev
RUN apt-get install -y mono-mcs mono-xbuild libmono-system-numerics4.0-cil libmono-microsoft-build-tasks-v4.0-4.0-cil
RUN apt-get install -y fftw3-dev
RUN apt-get install -y libhwloc-dev
RUN apt-get install -y ocl-icd-opencl-dev ocl-icd-libopencl1
RUN apt-get install -y fglrx fglrx-dev opencl-headers
RUN apt-get install -y freeglut3-dev libxmu-dev libxi-dev

# Copy and build bohrium source files from 'context'
COPY . ../
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN make
RUN make install

# Test Suite
ENV PYTHONPATH /usr/lib/python2.7/site-packages
ENTRYPOINT echo $BH_STACK && python test/python/numpytest.py

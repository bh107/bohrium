FROM ubuntu:16.04
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>
RUN mkdir -p /bohrium/build
WORKDIR /bohrium/build

RUN apt-get -qq update > /dev/null

# Set the locale
RUN apt-get -qq install locales > /dev/null
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install dependencies
RUN apt-get -qq update > /dev/null
RUN apt-get -qq install wget unzip build-essential > /dev/null
RUN apt-get -qq install cmake swig python python-numpy python-dev cython > /dev/null
RUN apt-get -qq install libboost-serialization-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-regex-dev > /dev/null
RUN apt-get -qq install mono-mcs mono-xbuild libmono-system-numerics4.0-cil libmono-microsoft-build-tasks-v4.0-4.0-cil > /dev/null
RUN apt-get -qq install libopenblas-dev libclblas-dev liblapacke-dev > /dev/null
RUN apt-get -qq install fftw3-dev libgl1-mesa-dev > /dev/null
RUN apt-get -qq install python3 python3-numpy python3-dev cython3 > /dev/null
RUN apt-get -qq install python2.7-scipy python3-scipy > /dev/null
RUN apt-get -qq install python-matplotlib python3-matplotlib > /dev/null
RUN apt-get -qq install python-netcdf4 python3-netcdf4 > /dev/null
RUN apt-get -qq install python-pyopencl python3-pyopencl > /dev/null
RUN apt-get -qq install zlib1g-dev > /dev/null
# RUN apt-get -qq install valgrind gdb vim cgdb > /dev/null

# Install OpenCV 3
ADD https://github.com/opencv/opencv/archive/3.2.0.zip .
RUN unzip 3.2.0.zip
RUN mkdir -p opencv-3.2.0/build
WORKDIR opencv-3.2.0/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
RUN make install -j4
RUN ldconfig

# Install AMD SDK for OpenCL
# RUN wget -nv http://jenkins.choderalab.org/userContent/AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2; exit 0;
RUN mkdir -p /opt/amd_src
ADD AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2 /opt/amd_src
ENV OPENCL_HOME /opt/AMDAPPSDK-2.9-1
ENV OPENCL_LIBPATH /opt/AMDAPPSDK-2.9-1/lib/x86_64
RUN sh /opt/amd_src/AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh -- -s -a yes > /dev/null
ENV OpenCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
ENV OpenCL_INCPATH "/opt/AMDAPPSDK-2.9-1/include"
ENV LD_LIBRARY_PATH "$OpenCL_LIBPATH:$LD_LIBRARY_PATH"


# This is a Dockerfile for installing and testing Bohrium
# It is based on the image 'bohrium/ubuntu', which must be on docker hub or locally.
# Please make sure that the build "context" is pointing to the root of Bohrium source files
# e.g. 'docker build -t bohrium -f <path to this file> <path to bohrium source>'
# Then you can run 'docker run -t bohrium' to Bohrium test

FROM bohrium/ubuntu:15.10
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>
RUN mkdir -p /bohrium/build
WORKDIR /bohrium/build

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Copy and build bohrium source files from "context"
COPY . ../
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF -DOpenACC=OFF
RUN make
RUN make install

# Test Suite
ENV PYTHONPATH /usr/lib/python2.7/site-packages
ENTRYPOINT echo $BH_STACK && python /bohrium/test/python/numpytest.py

FROM bohrium/ubuntu:16.04
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Download benchpress
RUN mkdir -p /benchpress
WORKDIR /benchpress/
ADD https://github.com/bh107/benchpress/archive/v2.0.zip ./benchpress.zip
RUN unzip -q benchpress.zip
RUN mv benchpress-* benchpress
ENV PATH "/benchpress/benchpress/bin:$PATH"
ENV PYTHONPATH "/benchpress/benchpress/module:$PYTHONPATH"

# Copy and build bohrium source files from "context"
RUN mkdir -p /bohrium/build
WORKDIR /bohrium/build
COPY . ../
RUN cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF -DUSE_WERROR=ON -DVEM_PROXY=ON
RUN make
RUN make install

# Test Suite
WORKDIR /bohrium

RUN echo "#/usr/bin/env bash" > test_exec.sh && \
    echo "shopt -s extglob" >> test_exec.sh  && \
    chmod +x test_exec.sh

ENTRYPOINT export PYTHONPATH="/usr/lib/$PY_EXEC/site-packages:$PYTHONPATH" && \
           echo "$TEST_EXEC" >> test_exec.sh                               ; \
           bash test_exec.sh

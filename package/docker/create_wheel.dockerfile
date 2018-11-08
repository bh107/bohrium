FROM bohrium/manylinux:latest
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>

# Copy bohrium source files from "context"
WORKDIR /bh
COPY . .

# The default build type is "Release"
ARG BUILD_TYPE=Release

RUN cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCORE_LINK_FLAGS='-static-libgcc -static-libstdc++' -DBoost_USE_STATIC_LIBS=ON -DVE_OPENMP_COMPILER_OPENMP_SIMD=OFF -DCYTHON_OPENMP=OFF -DEXT_VISUALIZER=OFF -DVEM_PROXY=OFF -DCMAKE_INSTALL_PREFIX=/bh/install -DFORCE_CONFIG_PATH=/bh/install -DCBLAS_LIBRARIES=/usr/lib64/atlas/libcblas.so.3 -DCBLAS_INCLUDES=/usr/include -DLAPACKE_LIBRARIES=/usr/lib64/atlas/liblapack.so.3 -DLAPACKE_INCLUDE_DIR=/usr/include/openblas -DPY_WHEEL=/opt/python/cp27-cp27mu/bin/python
RUN make install

# Build Bohrium with python2.7
ENV PATH /opt/python/cp27-cp27mu/bin/:$PATH
RUN bash /bh/build.sh 2.7
RUN bash /bh/wheel.sh 2.7
RUN pip install /bh/b2.7/dist/*
RUN pip install benchpress

# Build Bohrium with python3.4
ENV PATH /opt/python/cp34-cp34m/bin/:$PATH
RUN bash /bh/build.sh 3.4
RUN bash /bh/wheel.sh 3.4
RUN pip install /bh/b3.4/dist/*

# Build Bohrium with python3.5
ENV PATH /opt/python/cp35-cp35m/bin/:$PATH
RUN bash /bh/build.sh 3.5
RUN bash /bh/wheel.sh 3.5
RUN pip install /bh/b3.5/dist/*

# Build Bohrium with python3.6
ENV PATH /opt/python/cp36-cp36m/bin/:$PATH
RUN bash /bh/build.sh 3.6
RUN bash /bh/wheel.sh 3.6
RUN pip install /bh/b3.6/dist/*
RUN pip install benchpress

# Sanity check and info
RUN BH_STACK=opencl /opt/python/cp27-cp27mu/bin/python -m bohrium --info
RUN BH_STACK=opencl /opt/python/cp34-cp34m/bin/python -m bohrium --info
RUN BH_STACK=opencl /opt/python/cp35-cp35m/bin/python -m bohrium --info
RUN BH_STACK=opencl /opt/python/cp36-cp36m/bin/python -m bohrium --info

# Deploy script
WORKDIR /bh
RUN echo "#/usr/bin/env bash" > deploy.sh && \
    echo "python2.7 -m twine upload /bh/b*/dist/* || true" >> deploy.sh && \
    chmod +x deploy.sh

# Execute script
WORKDIR /bh
RUN echo "#/usr/bin/env bash" > exec.sh && \
    echo "shopt -s extglob" >> exec.sh  && \
    chmod +x exec.sh

# Run the command in `EXEC`
ENTRYPOINT echo "$EXEC" >> exec.sh && sleep 1 && cat exec.sh && ./exec.sh


FROM bohrium/manylinux:latest
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>

# Copy bohrium source files from "context"
WORKDIR /bh
COPY . .

# The default build type is "Release"
ARG BUILD_TYPE=Release

# List of Python version we want to build
ARG PY_VER_LIST="cp27-cp27mu;cp36-cp36m;cp37-cp37m;cp38-cp38;cp39-cp39"

# Script that creates links to the different python binaries
RUN echo $'#!/bin/bash\n\
IFS=";"\n\
for name in $1; do\n\
  ln -s /opt/python/${name}/bin/python /usr/bin/${name}\n\
done' > /bh/py_exe_links
RUN bash /bh/py_exe_links ${PY_VER_LIST}
RUN ls -l /usr/bin/cp*

# Build bohrium
RUN mkdir /bh/build
WORKDIR /bh/build
RUN AMDAPPSDKROOT=/opt/AMDAPP/ cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCORE_LINK_FLAGS='-static-libgcc -static-libstdc++' -DBoost_USE_STATIC_LIBS=ON -DBRIDGE_NPBACKEND=OFF -DVE_OPENMP_COMPILER_OPENMP_SIMD=OFF -DEXT_VISUALIZER=OFF -DVEM_PROXY=OFF -DCMAKE_INSTALL_PREFIX=/bh/install -DFORCE_CONFIG_PATH=/bh/install -DCBLAS_LIBRARIES=/usr/lib64/atlas/libcblas.so.3 -DCBLAS_INCLUDES=/usr/include -DLAPACKE_LIBRARIES=/usr/lib64/atlas/liblapack.so.3 -DLAPACKE_INCLUDE_DIR=/usr/include/openblas -DPY_WHEEL=/bh/wheel -DPY_EXE_LIST=$PY_VER_LIST
RUN make -j2
RUN make install

# Patch auditwheel to ignore libraries not found e.g. libcuda
RUN sed -i -e 's/if src_path is None:/if src_path is None:\n                    continue/g' /opt/_internal/tools/lib/python3.7/site-packages/auditwheel/repair.py

# Export the Bohrium C bridge and OpenCl.so
ENV LD_LIBRARY_PATH /bh/build/bridge/c/:/opt/AMDAPP/lib/x86_64/:$LD_LIBRARY_PATH

# Let's write a script that for each python version builds a manylinux2010 package of
# bohrium_api, install it along with bohrium, and runs a sanity check.
RUN echo $'#!/bin/bash\n\
IFS=";"\n\
for name in $PY_VER_LIST; do\n\
  export USE_CYTHON=1\n\
  auditwheel repair /bh/wheel/bohrium_api-*-${name}-*.whl -w /bh/wheelhouse\n\
  ${name} -m pip install /bh/wheelhouse/bohrium_api-*-${name}-*.whl\n\
  ${name} -m pip install cython numpy\n\
  ${name} -m pip install /bh/bridge/npbackend\n\
  ${name} -m pip install /bh/bridge/bh107\n\
  BH_STACK=opencl ${name} -m bohrium --info\n\
done\n\
cd /bh/bridge/npbackend/\n\
cp27-cp27mu setup.py build_ext\n\
unset USE_CYTHON\n\
cp /bh/README.rst .\n\
cp27-cp27mu setup.py sdist -d /bh/sdisthouse/\n\
cd /bh/bridge/bh107/\n\
cp27-cp27mu setup.py sdist -d /bh/sdisthouse/\n\
' > /bh/install.sh

# Let's run the install script
RUN bash /bh/install.sh

# Deploy script
WORKDIR /bh
RUN echo "#!/usr/bin/env bash" > deploy.sh && \
    echo "cp27-cp27mu -m twine upload /bh/wheelhouse/*.whl /bh/sdisthouse/*" >> deploy.sh && \
    chmod +x deploy.sh

# Execute script
WORKDIR /bh
RUN echo "#!/usr/bin/env bash" > exec.sh && \
    echo "shopt -s extglob" >> exec.sh  && \
    chmod +x exec.sh

# Run the command in `EXEC`
ENTRYPOINT echo "$EXEC" >> exec.sh && sleep 1 && cat exec.sh && ./exec.sh


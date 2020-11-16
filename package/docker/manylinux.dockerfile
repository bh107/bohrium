FROM quay.io/pypa/manylinux2010_x86_64:latest
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>

# The build root
RUN mkdir /b
WORKDIR /b

# Install Boost with -fPIC (system-wide)
ADD https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz .
RUN tar -xzf boost_1_66_0.tar.gz
WORKDIR /b/boost_1_66_0
RUN ./bootstrap.sh --without-libraries=python --without-libraries=mpi --with-icu
RUN ./b2 -q variant=release debug-symbols=off threading=multi runtime-link=shared  link=static,shared toolset=gcc cxxflags="-fPIC" --layout=system  install
WORKDIR /b

# Use Python 2.7 for the reset of the installation
ENV PATH /opt/python/cp27-cp27mu/bin/:$PATH

# Install AMD SDK for OpenCL (/opt/AMDAPPSDK-2.9-1)
RUN yum install -y redhat-lsb xz
RUN mkdir -p /b/amd_sdk
WORKDIR /b/amd_sdk
ADD https://github.com/ghostlander/AMD-APP-SDK/releases/download/v2.9.1/AMD-APP-SDK-v2.9-lnx64.tar .
RUN tar -xf AMD-APP-SDK-v2.9-lnx64.tar
RUN ls
WORKDIR /b/amd_sdk/AMD-APP-SDK-v2.9-lnx64
RUN sh Install-AMD-APP.sh
ENV OPENCL_HOME /opt/AMDAPP
ENV OPENCL_LIBPATH /opt/AMDAPP/lib/x86_64
WORKDIR /b

# Install BLAS/LAPACK extmethod dependencies
RUN yum install -y atlas-devel
RUN yum install -y openblas-devel

# Install OpenCV 3
WORKDIR /b
ADD https://github.com/opencv/opencv/archive/3.2.0.zip .
RUN unzip 3.2.0
RUN mkdir -p opencv-3.2.0/build
WORKDIR /b/opencv-3.2.0/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_LAPACK=OFF -DBUILD_SHARED_LIBS=NO -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++"
RUN make install -j4
RUN ldconfig

# Install GNU's libsigsegv
RUN yum install -y libsigsegv-devel

# Install Python build and depoly dependencies
RUN pip install virtualenv twine

# Install CUDA toolkit
WORKDIR /b
ADD cuda_8.0.44_linux.run .
RUN sh cuda_8.0.44_linux.run --silent --toolkit --override
WORKDIR /usr/local/cuda/
RUN rm -Rf lib64/*.a libnsight  libnvvp  nvml  nvvm  pkgconfig  \
           share  src samples tools jre extras doc

# Clean up
WORKDIR /
RUN rm -Rf /b
RUN yum clean all
RUN rm -Rf ~/.cache/pip

# Remove some unneeded AMD files
RUN rm -Rf /opt/AMDAPP/samples
RUN rm -Rf /opt/AMDAPP/lib/x86/

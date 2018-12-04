FROM quay.io/pypa/manylinux1_x86_64:latest
MAINTAINER Mads R. B. Kristensen <madsbk@gmail.com>

# cmake 2.8 is installed but need a link
RUN ln -s /usr/bin/cmake28 /usr/bin/cmake

# The build root
RUN mkdir /b
WORKDIR /b

# Install gcc7 (/opt/gcc7)
ADD http://mirrors.concertpass.com/gcc/releases/gcc-7.3.0/gcc-7.3.0.tar.gz .
RUN tar -xzf gcc-7.3.0.tar.gz
WORKDIR gcc-7.3.0
RUN ./contrib/download_prerequisites
RUN mkdir -p /b/gcc_build
WORKDIR /b/gcc_build
RUN /b/gcc-7.3.0/configure --prefix /opt/gcc7 --enable-languages=c --disable-bootstrap --disable-multilib
RUN make -j4
RUN make install
ENV PATH /opt/gcc7/bin:$PATH
ENV LD_LIBRARY_PATH "/opt/gcc7/lib64:$LD_LIBRARY_PATH"

# Install Boost with -fPIC (system-wide)
ADD https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz .
RUN tar -xzf boost_1_66_0.tar.gz
WORKDIR boost_1_66_0
RUN ./bootstrap.sh --without-libraries=python --without-libraries=mpi --with-icu
RUN ./b2 -q variant=release debug-symbols=off threading=multi runtime-link=shared  link=static,shared toolset=gcc cxxflags="-fPIC" --layout=system  install
WORKDIR /b

# Use Python 2.7 for the reset of the installation
ENV PATH /opt/python/cp27-cp27mu/bin/:$PATH

# Install AMD SDK for OpenCL (/opt/AMDAPPSDK-2.9-1)
RUN yum install -y redhat-lsb
RUN mkdir -p amd_src
ADD AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh amd_src
ENV OPENCL_HOME /opt/AMDAPPSDK-2.9-1
ENV OPENCL_LIBPATH /opt/AMDAPPSDK-2.9-1/lib/x86_64
RUN sh amd_src/AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh -- -s -a yes > /dev/null
ENV OpenCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
ENV OpenCL_INCPATH "/opt/AMDAPPSDK-2.9-1/include"
ENV LD_LIBRARY_PATH "$OpenCL_LIBPATH:$LD_LIBRARY_PATH"
WORKDIR /b

# Install BLAS/LAPACK extmethod dependencies
RUN yum install -y atlas-devel-3.8.3-1.el5.x86_64
RUN yum install -y openblas-devel-0.2.18-5.el5.x86_64

# Install OpenCV 3
WORKDIR /b
ADD https://github.com/opencv/opencv/archive/3.2.0.zip .
RUN unzip 3.2.0
RUN mkdir -p opencv-3.2.0/build
WORKDIR opencv-3.2.0/build
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
RUN rm -Rf /opt/AMDAPPSDK-2.9-1/samples
RUN rm -Rf /opt/AMDAPPSDK-2.9-1/lib/x86/

# Stripping unneeded data in the GCC binaries
RUN find /opt/gcc7/libexec/gcc/x86_64-pc-linux-gnu/7.3.0 -maxdepth 1 -type f -size +10M -print0 | xargs -0 strip --strip-unneeded --remove-section=.comment --remove-section=.note

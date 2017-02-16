#!/bin/bash

mkdir build
cd build
pwd
cmake .. -DCMAKE_BUILD_TYPE=Release -DBRIDGE_CIL=OFF -DVEM_PROXY=OFF -DEXT_BLAS=OFF -DEXT_CLBLAS=OFF -DVE_OPENCL=OFF -DEXT_VISUALIZER=OFF -DBoost_USE_STATIC_LIBS=ON -DCORE_LINK_FLAGS="-static-libgcc -static-libstdc++" -DCMAKE_INSTALL_PREFIX=$PREFIX

make VERBOSE=0
make install

# Set the OpenMP compiler command to "cc"
sed -ie 's/\/usr\/local\/bin\/cc/cc/g' $PREFIX/etc/bohrium/config.ini

# Turn off the OpenMP SIMD flag
sed -ie 's/-fopenmp-simd//g' $PREFIX/etc/bohrium/config.ini
sed -ie 's/compiler_openmp_simd = true/compiler_openmp_simd = false/g' $PREFIX/etc/bohrium/config.ini


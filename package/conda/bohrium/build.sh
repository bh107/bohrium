#!/bin/bash

mkdir build
cd build
export
ls $CONDA_PREFIX
ls $CONDA_PREFIX/lib
ls $CONDA_PREFIX/include
ls $PREFIX
ls $PREFIX/lib
ls $CONDA_PREFIX
ls
echo "Build Boost"
wget https://downloads.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.gz
tar -xzf boost_1_63_0.tar.gz
cd boost_1_63_0
bash $RECIPE_DIR/build_boost.sh $PWD/../boost
cd ..
ls $PWD/boost

cmake --version
cmake .. -DCMAKE_BUILD_TYPE=Release \
	 -DBRIDGE_CIL=OFF \
	 -DVEM_PROXY=OFF \
 	 -DEXT_CLBLAS=OFF \
	 -DVE_OPENCL=OFF \
	 -DEXT_VISUALIZER=OFF \
         -DBoost_NO_SYSTEM_PATHS=ON \
	 -DBOOST_ROOT=$PWD/boost \
	 -DBoost_USE_STATIC_LIBS=ON \
	 -DLAPACKE_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so \
	 -DCORE_LINK_FLAGS="-static-libgcc -static-libstdc++" \
	 -DFORCE_CONFIG_PATH=$PREFIX/etc/bohrium \
         -DLIBDIR=lib \
         -DPYTHON_EXECUTABLE=$PYTHON \
         -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib \
         -DCMAKE_INCLUDE_PATH=$CONDA_PREFIX/include \
	 -DCMAKE_INSTALL_PREFIX=$PREFIX

make VERBOSE=1
make install

# Set the OpenMP compiler command to "cc"
#sed -ie 's/compiler_cmd =.*$/compiler_cmd = cc/g' $PREFIX/etc/bohrium/config.ini

# Turn off the OpenMP SIMD flag
sed -ie 's/-fopenmp-simd//g' $PREFIX/etc/bohrium/config.ini
sed -ie 's/compiler_openmp_simd = true/compiler_openmp_simd = false/g' $PREFIX/etc/bohrium/config.ini


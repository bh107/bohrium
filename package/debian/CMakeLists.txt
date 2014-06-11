cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

project(BOHRIUM)
include(FeatureSummary)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Some Bohrium needs Python when building.")

#core modules
add_subdirectory(../../core ${CMAKE_CURRENT_BINARY_DIR}/core)
add_subdirectory(../../vem/node ${CMAKE_CURRENT_BINARY_DIR}/vem/node)
add_subdirectory(../../vem/proxy ${CMAKE_CURRENT_BINARY_DIR}/vem/proxy)
add_subdirectory(../../filter/pprint ${CMAKE_CURRENT_BINARY_DIR}/filter/pprint)
add_subdirectory(../../extmethods/matmul ${CMAKE_CURRENT_BINARY_DIR}/extmethods/matmul)
add_subdirectory(../../bridge/cpp ${CMAKE_CURRENT_BINARY_DIR}/bridge/cpp)
add_subdirectory(../../bridge/c ${CMAKE_CURRENT_BINARY_DIR}/bridge/c)
add_subdirectory(../../ve/cpu ${CMAKE_CURRENT_BINARY_DIR}/ve/cpu)

#optional modules
add_subdirectory(../../bridge/bhpy ${CMAKE_CURRENT_BINARY_DIR}/bridge/bhpy)
add_subdirectory(../../ve/gpu ${CMAKE_CURRENT_BINARY_DIR}/ve/gpu)
add_subdirectory(../../vem/cluster ${CMAKE_CURRENT_BINARY_DIR}/vem/cluster)
add_subdirectory(../../extmethods/fftw ${CMAKE_CURRENT_BINARY_DIR}/extmethods/fftw)

install(DIRECTORY ../../thirdparty/Random123/include/Random123 DESTINATION include COMPONENT bohrium)
install(DIRECTORY ../../test DESTINATION share/bohrium COMPONENT bohrium)
install(DIRECTORY ../../benchmark DESTINATION share/bohrium COMPONENT bohrium)

#Install the config file, which depends on global or local installation
configure_file(../../config.ini.in config.ini)
install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium COMPONENT bohrium)

MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)

#### NB: changes above this line only has effect when committed in git ####

#Make sure we only run the package script once
if(DEB_SRC_PPA_ONCE)
    return()
endif()

#Package setup
set(CPACK_GENERATOR "DEB")
execute_process(COMMAND git describe --tags --long WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE CPACK_PACKAGE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
#Lets remove the 'v' char in the version tag (e.g. v0.2-0-g6a2352d => 0.2-0-g6a2352d)
string(SUBSTRING ${CPACK_PACKAGE_VERSION} 1 -1 CPACK_PACKAGE_VERSION)

set(CPACK_PACKAGE_CONTACT "Bohrium Builder <builder@bh107.org>" CACHE STRING "The package contact info which is also used for signing the package")
set(CPACK_PACKAGE_DESCRIPTION "Bohrium Runtime System")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python")
set(CPACK_PACKAGE_NAME "Bohrium")
set(CPACK_PACKAGE_VENDOR "http://www.bh107.org")

#Debian specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libctemplate-dev, build-essential, libboost-dev, python (>= 2.7), python-numpy (>= 1.6), fftw3, mpich2, libmpich2-dev, opencl-dev, libopencl1, libgl-dev")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_DISTRIBUTION_NAME "ubuntu")
set(CPACK_DEBIAN_DISTRIBUTION_RELEASES trusty saucy)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../../COPYING")
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "ipython")

set(CPACK_DEBIAN_BUILD_DEPENDS python-numpy debhelper cmake swig libctemplate-dev libboost-dev python-cheetah python-dev fftw3-dev mpich2 libmpich2-dev ocl-icd-opencl-dev libgl-dev)

set(CPACK_COMPONENT_bohrium-gpu_DEPENDS "bohrium, opencl-dev, libopencl1, libgl-dev")
set(CPACK_COMPONENT_bohrium-gpu_DESCRIPTION_SUMMARY "The GPU component")

include(CPack)
message("CPACK_PACKAGE_VERSION: ${CPACK_PACKAGE_VERSION}")
message(STATUS "CPACK_COMPONENTS_ALL: ${CPACK_COMPONENTS_ALL}")
include(DebSourcePPA)

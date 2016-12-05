cmake_minimum_required(VERSION 2.8)

set(VE_OPENCL true CACHE BOOL "VE-OPENCL: Build the OpenCL code generator.")
if(NOT VE_OPENCL)
    return()
endif()

find_package(OpenCL)
set_package_properties(OpenCL PROPERTIES DESCRIPTION "Open Computing Language" URL "www.khronos.org/opencl/")
set_package_properties(OpenCL PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the GPU-VE.")

find_package(OpenGL)
set_package_properties(OpenGL PROPERTIES DESCRIPTION "Open Graphics Library" URL "www.opengl.org")
set_package_properties(OpenGL PROPERTIES TYPE RECOMMENDED PURPOSE "Required by OpenCL/GPU-VE.")

if(NOT (OPENCL_FOUND AND OPENGL_FOUND))
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${OPENCL_INCLUDE_DIRS})

file(GLOB SRC *.cpp)

add_library(bh_ve_opencl SHARED ${SRC})

target_link_libraries(bh_ve_opencl bh ${OPENCL_LIBRARIES})

install(TARGETS bh_ve_opencl DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium-opencl)

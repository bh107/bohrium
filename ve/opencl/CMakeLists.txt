cmake_minimum_required(VERSION 2.8)

set(VE_OPENCL true CACHE BOOL "VE-OPENCL: Build the OpenCL code generator.")
if(NOT VE_OPENCL)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_ve_opencl SHARED ${SRC})

target_link_libraries(bh_ve_opencl bh)

install(TARGETS bh_ve_opencl DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

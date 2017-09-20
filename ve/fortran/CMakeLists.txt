cmake_minimum_required(VERSION 2.8)

set(VE_FORTRAN true CACHE BOOL "VE-FORTRAN: Build the FORTRAN code generator.")
if(NOT VE_FORTRAN)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_ve_fortran SHARED ${SRC})

target_link_libraries(bh_ve_fortran bh)

install(TARGETS bh_ve_fortran DESTINATION ${LIBDIR} COMPONENT bohrium)

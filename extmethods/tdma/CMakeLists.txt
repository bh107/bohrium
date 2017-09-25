cmake_minimum_required(VERSION 2.8)

set(EXT_TDMA true CACHE BOOL "EXT-TDMA: Build TDMA solver extension method.")
if(NOT EXT_TDMA)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_library(bh_tdma SHARED main.cpp)

target_link_libraries(bh_tdma bh)

find_package(OpenMP COMPONENTS bohrium)
if(OpenMP_CXX_FOUND)
    set_target_properties(bh_tdma PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS} LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

install(TARGETS bh_tdma DESTINATION ${LIBDIR} COMPONENT bohrium)

# Add TDMA to OpenMP libs
set(BH_OPENMP_LIBS ${BH_OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_tdma${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)

cmake_minimum_required(VERSION 2.8)
set(FUSER_TOPOLOGICAL true CACHE BOOL "FUSER-TOPOLOGICAL: Build TOPOLOGICAL fuser.")
if(NOT FUSER_TOPOLOGICAL)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_library(bh_fuser_topological SHARED main.cpp)

#We depend on bh.so
target_link_libraries(bh_fuser_topological bh)

install(TARGETS bh_fuser_topological DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)


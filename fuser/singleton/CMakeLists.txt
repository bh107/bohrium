cmake_minimum_required(VERSION 2.8)
set(FUSER_SINGLETON true CACHE BOOL "FUSER-SINGLETON: Build SINGLETON fuser.")
if(NOT FUSER_SINGLETON)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_library(bh_fuser_singleton SHARED main.cpp)

#We depend on bh.so
target_link_libraries(bh_fuser_singleton bh)

install(TARGETS bh_fuser_singleton DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)


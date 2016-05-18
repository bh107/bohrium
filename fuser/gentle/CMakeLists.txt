cmake_minimum_required(VERSION 2.8)
set(FUSER_GENTLE true CACHE BOOL "FUSER-GENTLE: Build gentle fuser.")
if(NOT FUSER_GENTLE)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_library(bh_fuser_gentle SHARED main.cpp)

#We depend on bh.so
target_link_libraries(bh_fuser_gentle bh)

install(TARGETS bh_fuser_gentle DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

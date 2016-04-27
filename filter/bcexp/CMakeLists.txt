cmake_minimum_required(VERSION 2.8)
set(FILTER_BCEXP true CACHE BOOL "FILTER-BCEXP: Build the BCEXP filter.")
if(NOT FILTER_BCEXP)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_filter_bcexp SHARED ${SRC})

target_link_libraries(bh_filter_bcexp bh)    #We depend on bh.so

install(TARGETS bh_filter_bcexp DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)
set(FILTER_NONEREMOVER true CACHE BOOL "FILTER-NONEREMOVER: Build the NONEREMOVER filter.")
if(NOT FILTER_NONEREMOVER)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC main.cpp)

add_library(bh_filter_noneremover SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_noneremover bh)

install(TARGETS bh_filter_noneremover DESTINATION ${LIBDIR} COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)
set(FILTER_NONEREMOVER true CACHE BOOL "FILTER-NONEREMOVER: Build the NONEREMOVER filter.")
if(NOT FILTER_NONEREMOVER)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_noneremover SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_noneremover bh)

install(TARGETS bh_filter_noneremover DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)
set(FILTER_DIMCLEAN true CACHE BOOL "FILTER-DIMCLEAN: Build the DIMCLEAN filter.")
if(NOT FILTER_DIMCLEAN)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_dimclean SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_dimclean bh)

install(TARGETS bh_filter_dimclean DESTINATION lib COMPONENT bohrium)

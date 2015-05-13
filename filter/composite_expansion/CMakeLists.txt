cmake_minimum_required(VERSION 2.8)
set(FILTER_COMPOSITE_EXPANSION true CACHE BOOL "FILTER-COMPOSITE_EXPANSION: Build the COMPOSITE_EXPANSION filter.")
if(NOT FILTER_COMPOSITE_EXPANSION)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_composite_expansion SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_composite_expansion bh)

install(TARGETS bh_filter_composite_expansion DESTINATION lib COMPONENT bohrium)

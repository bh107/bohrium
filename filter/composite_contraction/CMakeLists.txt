cmake_minimum_required(VERSION 2.8)
set(FILTER_COMPOSITE_CONTRACTION true CACHE BOOL "FILTER-COMPOSITE_CONTRACTION: Build the COMPOSITE_CONTRACTION filter.")
if(NOT FILTER_COMPOSITE_CONTRACTION)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_composite_contraction SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_composite_contraction bh)

install(TARGETS bh_filter_composite_contraction DESTINATION lib COMPONENT bohrium)

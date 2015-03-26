cmake_minimum_required(VERSION 2.8)
set(FILTER_COMPLETE_REDUCTION true CACHE BOOL "FILTER-COMPLETE_REDUCTION: Build the COMPLETE_REDUCTION filter.")
if(NOT FILTER_COMPLETE_REDUCTION)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_complete_reduction SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_complete_reduction bh)

install(TARGETS bh_filter_complete_reduction DESTINATION lib COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)
set(FILTER_PPRINT true CACHE BOOL "FILTER-PPRINT: Build the PPRINT filter.")
if(NOT FILTER_PPRINT)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_pprint SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_pprint bh)

install(TARGETS bh_filter_pprint DESTINATION lib COMPONENT bohrium)

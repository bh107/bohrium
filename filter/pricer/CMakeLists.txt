cmake_minimum_required(VERSION 2.8)
set(FILTER_PRICER true CACHE BOOL "FILTER-PRICER: Build the PRICER filter.")
if(NOT FILTER_PRICER)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_pricer SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_pricer bh)

install(TARGETS bh_filter_pricer DESTINATION lib COMPONENT bohrium)

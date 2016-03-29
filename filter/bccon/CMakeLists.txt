cmake_minimum_required(VERSION 2.8)
set(FILTER_BCCON true CACHE BOOL "FILTER-BCCON: Build the BCCON filter.")
if(NOT FILTER_BCCON)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_bccon SHARED ${SRC})

target_link_libraries(bh_filter_bccon bh)  # We depend on bh.so

install(TARGETS bh_filter_bccon DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)
set(FUSER_GREEDY true CACHE BOOL "FUSER-GREEDY: Build greedy fuser.")
if(NOT FUSER_GREEDY)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_library(bh_fuser_greedy SHARED main.cpp)

#We depend on bh.so
target_link_libraries(bh_fuser_greedy bh)

install(TARGETS bh_fuser_greedy DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

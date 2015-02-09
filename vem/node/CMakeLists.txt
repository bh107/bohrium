cmake_minimum_required(VERSION 2.8)
set(VEM_NODE true CACHE BOOL "VEM-NODE: Build the node VEM.")
if(NOT VEM_NODE)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_vem_node SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_vem_node bh)

install(TARGETS bh_vem_node DESTINATION lib COMPONENT bohrium)

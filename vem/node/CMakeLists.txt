cmake_minimum_required(VERSION 2.8)
include(../../CMakeLists.inc)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(bh_vem_node SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_vem_node bh)

install(TARGETS bh_vem_node DESTINATION lib)

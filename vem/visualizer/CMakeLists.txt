cmake_minimum_required(VERSION 2.8)
set(VEM_VISUALIZER true CACHE BOOL "VEM-VISUALIZER: Build the visualizer VEM.")
if(NOT VEM_VISUALIZER)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_vem_visualizer SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_vem_visualizer bh)

install(TARGETS bh_vem_visualizer DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

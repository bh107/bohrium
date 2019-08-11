cmake_minimum_required(VERSION 2.8)

add_subdirectory(pybind11)

pybind11_add_module(example main.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/bridge/cxx/include)
include_directories(${CMAKE_BINARY_DIR}/bridge/cxx/include)
target_link_libraries(example bhxx)
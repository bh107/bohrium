cmake_minimum_required(VERSION 2.8)
include(../../../CMakeLists.inc)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp)

add_executable(hello_world ${SRC})

#We depend on bh.so
target_link_libraries(hello_world bh)

#We need to include the header files of the C++ Bridge
include_directories(../bh)

set_target_properties(hello_world PROPERTIES COMPILE_FLAGS "-std=c++0x")

install(TARGETS hello_world DESTINATION bin)


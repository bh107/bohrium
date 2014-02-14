cmake_minimum_required(VERSION 2.8)
include_directories(../../../include)

file(GLOB SRC hello_world.cpp)

add_executable(bh_hello_world_cpp ${SRC})

#We depend on bh.so
target_link_libraries(bh_hello_world_cpp bh)

#We need to include the header files of the C++ Bridge
include_directories(../bh)

set_target_properties(bh_hello_world_cpp PROPERTIES COMPILE_FLAGS "-std=c++0x")

install(TARGETS bh_hello_world_cpp DESTINATION bin)


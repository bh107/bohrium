cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_matmul SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_matmul bh)

install(TARGETS bh_matmul DESTINATION lib)

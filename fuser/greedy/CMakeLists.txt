cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_fuser_greedy SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_fuser_greedy bh)

install(TARGETS bh_fuser_greedy DESTINATION lib COMPONENT bohrium)

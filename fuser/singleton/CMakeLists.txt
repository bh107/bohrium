cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_fuser_singleton SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_fuser_singleton bh)

install(TARGETS bh_fuser_singleton DESTINATION lib COMPONENT bohrium)

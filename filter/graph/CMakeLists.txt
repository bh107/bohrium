cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_graph SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_graph bh)

install(TARGETS bh_filter_graph DESTINATION lib COMPONENT bohrium)

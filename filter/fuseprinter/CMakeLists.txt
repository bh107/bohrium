cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_filter_fuseprinter SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_filter_fuseprinter bh)

install(TARGETS bh_filter_fuseprinter DESTINATION lib COMPONENT bohrium)

cmake_minimum_required(VERSION 2.8)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(iniparser SHARED ${SRC})

install(TARGETS iniparser DESTINATION lib COMPONENT bohrium)

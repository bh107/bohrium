cmake_minimum_required(VERSION 2.6)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

add_library(iniparser SHARED ${SRC})

install(TARGETS iniparser DESTINATION lib)

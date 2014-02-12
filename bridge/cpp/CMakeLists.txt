cmake_minimum_required(VERSION 2.8)
include(../../CMakeLists.inc)

#Install all header files
install(DIRECTORY bh DESTINATION include)

#Compile hellow world as a test
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)

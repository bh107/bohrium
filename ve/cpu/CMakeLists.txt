cmake_minimum_required(VERSION 2.8)
include(../../CMakeLists.inc)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/bh_ve_cpu.cpp)

add_library(bh_ve_cpu SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_ve_cpu bh)

set_target_properties(bh_ve_cpu PROPERTIES COMPILE_FLAGS "-std=c++0x")

install(TARGETS bh_ve_cpu DESTINATION lib)

#External dependencies
find_package(CTemplate REQUIRED)
include_directories(${CTEMPLATE_INCLUDE_DIR})
set(LIBS ${LIBS} ${CTEMPLATE_LIBRARIES})
target_link_libraries(bh_ve_cpu ${LIBS})


#We need to install some empty dirs
FILE(MAKE_DIRECTORY cpu)
FILE(MAKE_DIRECTORY cpu/kernels)
FILE(MAKE_DIRECTORY cpu/objects)
install(DIRECTORY cpu DESTINATION .)

#Install all kernal templates
install(DIRECTORY templates DESTINATION cpu FILES_MATCHING PATTERN "*.tpl")

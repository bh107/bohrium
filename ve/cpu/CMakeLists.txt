cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

file(GLOB SRC bh_ve_cpu.cpp)

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

#Install all kernal templates
install(DIRECTORY templates DESTINATION share/bh FILES_MATCHING PATTERN "*.tpl")

cmake_minimum_required(VERSION 2.6)

include_directories(../../include)
include_directories(../cpp/bh)
include_directories(codegen/output/)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c.h
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_type_definitions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_implementation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_implementation_basics.cpp)

file(GLOB DEPS codegen/templates/*.ctpl codegen/*.json codegen/*.py)
add_custom_command(
  OUTPUT ${SRC}
  COMMAND ./codegen/gen.py
  DEPENDS ${DEPS}  #TODO: gen.py writes a old timestamp thus this will always run
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(bhc SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bhc bh)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c.h DESTINATION include/bh)
install(TARGETS bhc DESTINATION lib)

#Compile hellow world as a test
add_subdirectory(examples)


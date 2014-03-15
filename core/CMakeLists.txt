cmake_minimum_required(VERSION 2.6)

include_directories(../include)
file(GLOB SRC *.cpp bhir/*.cpp)

#The timing require C++11
if(USE_CXX98)
    file(GLOB EXCLUDE_SRC bh_timing.cpp)
    list(REMOVE_ITEM SRC ${EXCLUDE_SRC})
endif()

add_library(bh SHARED ${SRC} bh_opcode.cpp)

#Rules for how to generate bh_opcode.cpp and bh_opcode.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bh_opcode.cpp COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py VERBATIM)
set_source_files_properties(bh_opcode.cpp PROPERTIES GENERATED TRUE)

#We depend on iniparser
add_subdirectory(iniparser)
target_link_libraries(bh iniparser)

#We depend on dlopen etc.
target_link_libraries(bh ${CMAKE_DL_LIBS})

install(TARGETS bh DESTINATION lib)


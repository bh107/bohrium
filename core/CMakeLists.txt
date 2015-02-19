cmake_minimum_required(VERSION 2.8)

include_directories(../include)
file(GLOB SRC *.cpp)

add_library(bh SHARED ${SRC} bh_opcode.cpp)

#Rules for how to generate bh_opcode.cpp and bh_opcode.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bh_opcode.cpp COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py VERBATIM)
set_source_files_properties(bh_opcode.cpp PROPERTIES GENERATED TRUE)

#We depend on iniparser
add_subdirectory(iniparser)
target_link_libraries(bh iniparser)

#We depend on dlopen etc.
target_link_libraries(bh ${CMAKE_DL_LIBS})

#We depend on boost
target_link_libraries(bh ${Boost_LIBRARIES})

install(TARGETS bh DESTINATION lib COMPONENT bohrium)
install(DIRECTORY ../include/ DESTINATION include/bohrium COMPONENT bohrium)


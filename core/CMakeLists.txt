cmake_minimum_required(VERSION 2.8)                     # cmake features

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC *.cpp)

add_library(bh SHARED ${SRC} bh_opcode.cpp)

# Rules for how to generate bh_opcode.cpp and bh_opcode.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bh_opcode.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/../include/bh_opcode.h
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py VERBATIM)
set_source_files_properties(bh_opcode.cpp PROPERTIES GENERATED TRUE)

add_subdirectory(iniparser)                     # bh_component depends on iniparser
target_link_libraries(bh iniparser)
target_link_libraries(bh ${CMAKE_DL_LIBS})      # bh_component depends on dlopen etc.
target_link_libraries(bh ${Boost_LIBRARIES})    # A shit ton of stuff depends on boost

install(TARGETS bh DESTINATION lib COMPONENT bohrium)
install(DIRECTORY ${BOHRIUM_SOURCE_DIR}/include/ DESTINATION include/bohrium COMPONENT bohrium)
